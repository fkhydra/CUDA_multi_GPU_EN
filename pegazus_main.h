#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "helper_cuda.h"
#include <d2d1.h>
#include <d2d1helper.h>
#pragma comment(lib, "d2d1")
#include <mmsystem.h>
#pragma comment(lib, "winmm.lib")

#include "pegazus_config.h"
#include "pegazus_2D.h"
#include "pegazus_3D.h"

ID2D1Factory* pD2DFactory = NULL;
ID2D1HwndRenderTarget* pRT = NULL;
ID2D1Bitmap* frameinram = NULL;
unsigned int D2D_imagebuffer[SCREEN_WIDTH * SCREEN_HEIGHT];

void PEGA_init(HWND hwnd);
void PEGA_free_Direct2D(void);
void PEGA_free2D(void);
void PEGA_clearscreen(void);
__global__ void PEGA_merge_CUDA_images(int maxcount, unsigned int* puffer, unsigned int* maskpuffer, unsigned int* copypuffer);
void PEGA_merge_down_2D_buffer(void);
void PEGA_swap_buffer(void);
void PEGA_render_2D();

void PEGA_create_HOST_2D_point_list(int count);
void PEGA_create_CUDA_2D_point_list(int count);
void PEGA_2D_point_reset(void);
void PEGA_add_2D_point(int x, int y, int color);
void PEGA_push_points_to_GPU(void);
__global__ void CUDA_render_2D_points(int maxcount, PEGA_SHAPE_2D_POINT* data, unsigned int* buffer, unsigned int* maskpuffer);

void PEGA_create_HOST_2D_line_list(int count);
void PEGA_create_CUDA_2D_line_list(int count);
void PEGA_2D_line_reset(void);
void PEGA_add_2D_line(int x1, int y1, int x2, int y2, int color);
void PEGA_push_lines_to_GPU(void);
__global__ void CUDA_render_2D_lines(int maxcount, PEGA_SHAPE_2D_LINE* data, unsigned int* buffer, unsigned int* maskpuffer);

void PEGA_add_2D_triangle(int x1, int y1, int x2, int y2, int x3, int y3, int color);
void PEGA_create_HOST_2D_triangle_list(int count);
void PEGA_create_CUDA_2D_triangle_list(int count);
void PEGA_2D_triangle_reset(void);
void PEGA_push_triangles_to_GPU(void);
__global__ void CUDA_render_2D_triangles(int maxcount, PEGA_SHAPE_2D_TRIANGLE* data, unsigned int* buffer, unsigned int* maskpuffer);

void PEGA_init(HWND hwnd)
{
 int i;
 cudaDeviceProp devprops;
 D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &pD2DFactory);
 pD2DFactory->CreateHwndRenderTarget(
  D2D1::RenderTargetProperties(),
  D2D1::HwndRenderTargetProperties(
   hwnd, D2D1::SizeU(SCREEN_WIDTH, SCREEN_HEIGHT)), &pRT);
 pRT->CreateBitmap(D2D1::SizeU(SCREEN_WIDTH, SCREEN_HEIGHT),
  D2D1::BitmapProperties(D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM,
   D2D1_ALPHA_MODE_IGNORE)), &frameinram);
 
 cudaGetDeviceCount(&CUDA_DEVICE_COUNT);
 PEGA_set_load_distribution_mode(PEGA_EVEN_DISTRIBUTION);
 //PEGA_set_load_distribution_mode(PEGA_SEQUENTIAL_DISTRIBUTION);
 //PEGA_limit_device_count(1);
 for (i = 0; i < CUDA_DEVICE_COUNT; ++i)
 {
  cudaSetDevice(i);
  cudaDeviceReset();
  cudaGetDeviceProperties(&devprops, i);
  PEGA_DEVICES.devinfo[i].cores = _ConvertSMVer2Cores(devprops.major,devprops.minor) * devprops.multiProcessorCount;
  PEGA_DEVICES.devinfo[i].memsize = (unsigned long long)devprops.totalGlobalMem/1024/1024/1024;//GB value
  cudaMalloc((void**)&CUDA_imagebuffer[i], SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(unsigned int));
  cudaMalloc((void**)&CUDA_maskimagebuffer[i], SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(unsigned int));
 } 
 cudaSetDevice(0);
 cudaMalloc((void**)&CUDA_DEV0_COPY_imagebuffer, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(unsigned int));
}

void PEGA_free_Direct2D(void)
{
 pRT->Release();
 pD2DFactory->Release();
}

void PEGA_free2D(void)
{
 int i;
 points2D_MAX = 0;
 free(points2D);
 free(lines2D);
 free(triangles2D);
 for (i = 0; i < CUDA_DEVICE_COUNT; ++i)
 {
  cudaSetDevice(i);
  cudaFree(CUDA_imagebuffer[i]);
  cudaFree(CUDA_maskimagebuffer[i]);
  cudaFree(CUDA_DEV0_COPY_imagebuffer);
  cudaFree(CUDA_points2D[i]);
  cudaFree(CUDA_lines2D[i]);
  cudaFree(CUDA_triangles2D[i]);
 }
}

__global__ void PEGA_merge_CUDA_images(int maxcount, unsigned int* puffer, unsigned int* maskpuffer, unsigned int* copypuffer)
{
 int i;
 int index = (blockIdx.x * blockDim.x) + threadIdx.x;
 int stride = blockDim.x * gridDim.x;
 for (i = index; i < maxcount; i += stride)
 {
  if (maskpuffer[i] == 1) puffer[i] = copypuffer[i];
 }
}

void PEGA_merge_down_2D_buffer(void)
{
 int i;
 cudaSetDevice(0);
 if (CUDA_DEVICE_COUNT > 1)
 {
  for (i = 1; i < CUDA_DEVICE_COUNT; ++i) //start with dev. #1 !
  {
   cudaMemcpyPeer(CUDA_maskimagebuffer[0], 0, CUDA_maskimagebuffer[i], i, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(unsigned int));
   cudaMemcpyPeer(CUDA_DEV0_COPY_imagebuffer, 0, CUDA_imagebuffer[i], i, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(unsigned int));
   PEGA_merge_CUDA_images << <CUDA_BLOCKS, CUDA_CORES >> > (SCREEN_WIDTH * SCREEN_HEIGHT, CUDA_imagebuffer[0], CUDA_maskimagebuffer[0], CUDA_DEV0_COPY_imagebuffer);
   cudaDeviceSynchronize();
  }
 }
}

void PEGA_swap_buffer(void)
{
 D2D1_RECT_U myarea;
 cudaMemcpy(D2D_imagebuffer, CUDA_imagebuffer[0], SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(unsigned int), cudaMemcpyDeviceToHost);
 myarea.left = 0;
 myarea.top = 0;
 myarea.right = SCREEN_WIDTH ;
 myarea.bottom = SCREEN_HEIGHT ;
 frameinram->CopyFromMemory(&myarea, D2D_imagebuffer, SCREEN_WIDTH * sizeof(unsigned int));
 pRT->BeginDraw();
 pRT->DrawBitmap(frameinram, D2D1::RectF(0.0f, 0.0f, SCREEN_WIDTH , SCREEN_HEIGHT ), 1.0f, D2D1_BITMAP_INTERPOLATION_MODE_NEAREST_NEIGHBOR, NULL);
 pRT->EndDraw();
}

void PEGA_create_HOST_2D_point_list(int count)
{
 points2D_MAX = count;
 points2D_counter = 0;
 points2D = (PEGA_SHAPE_2D_POINT*)malloc(count * sizeof(PEGA_SHAPE_2D_POINT));
}

void PEGA_create_HOST_2D_line_list(int count)
{
 lines2D_MAX = count;
 lines2D_counter = 0;
 lines2D = (PEGA_SHAPE_2D_LINE*)malloc(count * sizeof(PEGA_SHAPE_2D_LINE));
}

void PEGA_create_HOST_2D_triangle_list(int count)
{
 triangles2D_MAX = count;
 triangles2D_counter = 0;
 triangles2D = (PEGA_SHAPE_2D_TRIANGLE*)malloc(count * sizeof(PEGA_SHAPE_2D_TRIANGLE));
}

void PEGA_create_CUDA_2D_point_list(int count)
{
 int i;
 for (i = 0; i < CUDA_DEVICE_COUNT; ++i)
 {
  cudaSetDevice(i);
  cudaMalloc((void**)&CUDA_points2D[i], (count / CUDA_DEVICE_COUNT) * sizeof(PEGA_SHAPE_2D_POINT));
 }
}

void PEGA_create_CUDA_2D_line_list(int count)
{
 int i;
 for (i = 0; i < CUDA_DEVICE_COUNT; ++i)
 {
  cudaSetDevice(i);
  cudaMalloc((void**)&CUDA_lines2D[i], (count / CUDA_DEVICE_COUNT) * sizeof(PEGA_SHAPE_2D_LINE));
 }
}

void PEGA_create_CUDA_2D_triangle_list(int count)
{
 int i;
 for (i = 0; i < CUDA_DEVICE_COUNT; ++i)
 {
  cudaSetDevice(i);
  cudaMalloc((void**)&CUDA_triangles2D[i], (count / CUDA_DEVICE_COUNT) * sizeof(PEGA_SHAPE_2D_TRIANGLE));
 }
}

void PEGA_2D_point_reset(void)
{
 points2D_counter = 0;
}

void PEGA_2D_line_reset(void)
{
 lines2D_counter = 0;
}

void PEGA_2D_triangle_reset(void)
{
 triangles2D_counter = 0;
}

void PEGA_add_2D_point(int x, int y, int color)
{
 if (points2D_counter == points2D_MAX) return;
 if (x >= SCREEN_WIDTH || y >= SCREEN_HEIGHT || x < 0 || y < 0) return;
 points2D[points2D_counter].x1 = x;
 points2D[points2D_counter].y1 = y;
 points2D[points2D_counter++].color = color;
}

void PEGA_add_2D_line(int x1, int y1, int x2, int y2, int color)
{
 if (lines2D_counter == lines2D_MAX) return;
 if (x1 >= SCREEN_WIDTH) x1 = SCREEN_WIDTH-1;
 else if (x1 < 0) x1 = 0;
 if (y1 >= SCREEN_HEIGHT) y1 = SCREEN_HEIGHT-1;
 else if (y1 < 0) y1 = 0;
 if (x2 >= SCREEN_WIDTH) x2 = SCREEN_WIDTH-1;
 else if (x2 < 0) x2 = 0;
 if (y2 >= SCREEN_HEIGHT) y2 = SCREEN_HEIGHT-1;
 else if (y2 < 0) y2 = 0;
 lines2D[lines2D_counter].x1 = x1;
 lines2D[lines2D_counter].y1 = y1;
 lines2D[lines2D_counter].x2 = x2;
 lines2D[lines2D_counter].y2 = y2;
 lines2D[lines2D_counter++].color = color;
}

void PEGA_add_2D_triangle(int x1, int y1, int x2, int y2, int x3, int y3, int color)
{
 if ((triangles2D_counter) == triangles2D_MAX) return;
 if (x1 >= SCREEN_WIDTH) x1 = SCREEN_WIDTH - 1;
 else if (x1 < 0) x1 = 0;
 if (y1 >= SCREEN_HEIGHT) y1 = SCREEN_HEIGHT - 1;
 else if (y1 < 0) y1 = 0;
 if (x2 >= SCREEN_WIDTH) x2 = SCREEN_WIDTH - 1;
 else if (x2 < 0) x2 = 0;
 if (y2 >= SCREEN_HEIGHT) y2 = SCREEN_HEIGHT - 1;
 else if (y2 < 0) y2 = 0;
 if (x3 >= SCREEN_WIDTH) x3 = SCREEN_WIDTH - 1;
 else if (x3 < 0) x3 = 0;
 if (y3 >= SCREEN_HEIGHT) y3 = SCREEN_HEIGHT - 1;
 else if (y3 < 0) y3 = 0;
 triangles2D[triangles2D_counter].x1 = x1;
 triangles2D[triangles2D_counter].y1 = y1;
 triangles2D[triangles2D_counter].x2 = x2;
 triangles2D[triangles2D_counter].y2 = y2;
 triangles2D[triangles2D_counter].x3 = x3;
 triangles2D[triangles2D_counter].y3 = y3;
 triangles2D[triangles2D_counter++].color = color;
}

void PEGA_push_points_to_GPU(void)
{
 int i;
 
 if (points2D_counter > 0) {
  PEGA_calculate_load_distribution(points2D_counter, PEGA2DPOINT);
  for (i = 0; i < CUDA_DEVICE_COUNT; ++i)
  {
   if (PEGA_DEVICES.devinfo[i].work_items[PEGA2DPOINT] > 0)
   {
   cudaSetDevice(i);
   cudaMemcpy(CUDA_points2D[i], points2D + PEGA_DEVICES.devinfo[i].start_item[PEGA2DPOINT], PEGA_DEVICES.devinfo[i].work_items[PEGA2DPOINT] * sizeof(PEGA_SHAPE_2D_POINT), cudaMemcpyHostToDevice);
//   cudaMemcpy(CUDA_points2D[i], points2D + (i * points2D_counter / CUDA_DEVICE_COUNT ), points2D_counter / CUDA_DEVICE_COUNT * sizeof(PEGA_SHAPE_2D_POINT), cudaMemcpyHostToDevice);
   //cudaMemcpy(CUDA_points2D[i], points2D , points2D_counter * sizeof(PEGA_SHAPE_2D_POINT), cudaMemcpyHostToDevice);
   }   
  }
 }
}

void PEGA_push_lines_to_GPU(void)
{
 int i;
 if (lines2D_counter > 0) {
  PEGA_calculate_load_distribution(lines2D_counter, PEGA2DLINE);
  for (i = 0; i < CUDA_DEVICE_COUNT; ++i)
  {
   if (PEGA_DEVICES.devinfo[i].work_items[PEGA2DLINE] > 0)
   {
    cudaSetDevice(i);
    cudaMemcpy(CUDA_lines2D[i], lines2D + PEGA_DEVICES.devinfo[i].start_item[PEGA2DLINE], PEGA_DEVICES.devinfo[i].work_items[PEGA2DLINE] * sizeof(PEGA_SHAPE_2D_LINE), cudaMemcpyHostToDevice);
    //cudaMemcpy(CUDA_lines2D[i], lines2D + (i * lines2D_counter / CUDA_DEVICE_COUNT), lines2D_counter / CUDA_DEVICE_COUNT * sizeof(PEGA_SHAPE_2D_LINE), cudaMemcpyHostToDevice);
   }   
  }
 }
}

void PEGA_push_triangles_to_GPU(void)
{
 int i;
 if (triangles2D_counter > 0) {
  PEGA_calculate_load_distribution(triangles2D_counter, PEGA2DTRIANGLE);
  for (i = 0; i < CUDA_DEVICE_COUNT; ++i)
  {
   if (PEGA_DEVICES.devinfo[i].work_items[PEGA2DTRIANGLE] > 0)
   {
    cudaSetDevice(i);
    cudaMemcpy(CUDA_triangles2D[i], triangles2D + PEGA_DEVICES.devinfo[i].start_item[PEGA2DTRIANGLE], PEGA_DEVICES.devinfo[i].work_items[PEGA2DTRIANGLE] * sizeof(PEGA_SHAPE_2D_TRIANGLE), cudaMemcpyHostToDevice);
    //cudaMemcpy(CUDA_triangles2D[i], triangles2D + (i * triangles2D_counter / CUDA_DEVICE_COUNT), triangles2D_counter / CUDA_DEVICE_COUNT * sizeof(PEGA_SHAPE_2D_TRIANGLE), cudaMemcpyHostToDevice);
   }
  }
 }
}

void PEGA_render_2D()
{
 int i;
 int blockSize = CUDA_CORES;
 int numBlocks;
 if (points2D_counter > 0)
 {
  numBlocks = (points2D_counter + blockSize - 1) / blockSize;
  for (i = 0; i < CUDA_DEVICE_COUNT; ++i)
  {
   cudaSetDevice(i);
   CUDA_render_2D_points << <numBlocks, blockSize >> > (PEGA_DEVICES.devinfo[i].work_items[PEGA2DPOINT], CUDA_points2D[i], CUDA_imagebuffer[i], CUDA_maskimagebuffer[i]);
   //CUDA_render_2D_points << <CUDA_BLOCKS, CUDA_CORES >> > (points2D_counter , CUDA_points2D[i], CUDA_imagebuffer[i], CUDA_maskimagebuffer[i]);
  }
  cudaDeviceSynchronize();
 }
 if (lines2D_counter > 0)
 {
  numBlocks = (lines2D_counter + blockSize - 1) / blockSize;
  for (i = 0; i < CUDA_DEVICE_COUNT; ++i)
  {
   cudaSetDevice(i);
   CUDA_render_2D_lines << <numBlocks, blockSize >> > (PEGA_DEVICES.devinfo[i].work_items[PEGA2DLINE], CUDA_lines2D[i], CUDA_imagebuffer[i], CUDA_maskimagebuffer[i]);
  }
  cudaDeviceSynchronize();
 }
 if (triangles2D_counter > 0)
 {
  numBlocks = (triangles2D_counter + blockSize - 1) / blockSize;
  for (i = 0; i < CUDA_DEVICE_COUNT; ++i)
  {
   cudaSetDevice(i);
   CUDA_render_2D_triangles << <numBlocks, blockSize >> > (PEGA_DEVICES.devinfo[i].work_items[PEGA2DTRIANGLE], CUDA_triangles2D[i], CUDA_imagebuffer[i], CUDA_maskimagebuffer[i]);
  }
  cudaDeviceSynchronize();
 }
}

void PEGA_clearscreen(void)
{
 int i;

 for (i = 0; i < CUDA_DEVICE_COUNT; ++i)
 {
  cudaSetDevice(i);
  cudaMemset(CUDA_imagebuffer[i], 255, SCREEN_HEIGHT * SCREEN_WIDTH * sizeof(unsigned int));
  cudaMemset(CUDA_maskimagebuffer[i], 0, SCREEN_HEIGHT * SCREEN_WIDTH * sizeof(unsigned int));
 }
}

__global__ void CUDA_render_2D_points(int maxcount, PEGA_SHAPE_2D_POINT* data, unsigned int* buffer, unsigned int* maskpuffer)
{
 int i;
 int index = (blockIdx.x * blockDim.x) + threadIdx.x;
 int stride = blockDim.x * gridDim.x;
 for (i = index; i < maxcount; i += stride)
 {
  CUDA_SetPixel2D(data[i].x1, data[i].y1, data[i].color, buffer, maskpuffer);
 }
}

__global__ void CUDA_render_2D_lines(int maxcount, PEGA_SHAPE_2D_LINE* data, unsigned int* buffer, unsigned int* maskpuffer)
{
 int i;
 int index = (blockIdx.x * blockDim.x) + threadIdx.x;
 int stride = blockDim.x * gridDim.x;
 for (i = index; i < maxcount; i += stride)
 {
  CUDA_DrawLine2D(data[i].x1, data[i].y1, data[i].x2, data[i].y2, data[i].color, buffer, maskpuffer);
 }
}

__global__ void CUDA_render_2D_triangles(int maxcount, PEGA_SHAPE_2D_TRIANGLE* data, unsigned int* buffer, unsigned int* maskpuffer)
{
 int i;
 int index = (blockIdx.x * blockDim.x) + threadIdx.x;
 int stride = blockDim.x * gridDim.x;
 for (i = index; i < maxcount; i += stride)
 {
  CUDA_FillTriangle(data[i].x1, data[i].y1, data[i].x2, data[i].y2, data[i].x3, data[i].y3, data[i].color, buffer, maskpuffer);
 }
}
