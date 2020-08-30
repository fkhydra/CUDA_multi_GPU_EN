#pragma once
#include "pegazus_config.h"

Vec3f vLight;
bool enable_lighting;

float* CUDA_Zbuffer[CUDA_MAX_DEVICES], * CUDA_DEV0_COPY_Zbuffer;

float Math_PI = 3.14159265358979323846;

float zoom_value;
int render_inprogress;
float persp_deg;

Vec3f rot_deg, rot_deg2;
Vec3f* raw_vertices, * CUDA_raw_vertices[CUDA_MAX_DEVICES], * CUDA_rotated_vertices[CUDA_MAX_DEVICES];
int raw_vertices_length, MAX_3d_vertex_count;

//*******for measurements********
long int counter_vert, counter_poly;
float fps_stat;
int bmark_beginning;
int bmark_ending;

void PEGA_init_3D(void);
void PEGA_3D_vertex_reset(void);
void PEGA_create_HOST_CUDA_3D_vertex_list(int count);
void PEGA_free3D(void);
void PEGA_rotate_3D(void);
void PEGA_render_3D(void);
void PEGA_merge_down_zbuffers(void);
__global__ void CUDA_Merge_Zbuffers(float* main_zpuffer, float* secondary_zpuffer, unsigned int* main_imagedata, unsigned int* secondary_imagedata);
__global__ void PEGA_3D_rotation(int maxitemcount, Vec3f* rawarray, Vec3f* rotarray, Vec3f deg_cos, Vec3f deg_sin);
void PEGA_add_3D_triangle(int x1, int y1, int z1, int x2, int y2, int z2, int x3, int y3, int z3);
void PEGA_push_3D_triangles_to_GPU(void);
void PEGA_clear_zbuffer(void);
__global__ void PEGA_cleanup_zbuffer(float* zpuffer);
__global__ void PEGA_render_objects(int maxitemcount, Vec3f* rotarray, unsigned int* puffer, float* zpuffer, Vec3f lightvektor, bool islightenabled);
__device__ void CUDA_SetPixel_Zbuffer(int x1, int y1, int z1, int mycolor, unsigned int* puffer, float* zpuffer);
__device__ void CUDA_DrawLine_Zbuffer(int x1, int y1, int z1, int x2, int y2, int z2, int mycolor, unsigned int* puffer, float* zpuffer);
__device__ void CUDA_FillTriangle_Zbuffer(int x1, int y1, int z1, int x2, int y2, int z2, int x3, int y3, int z3, int mycolor, unsigned int* puffer, float* zpuffer);
void PEGA_zoom_in(void);
void PEGA_zoom_out(void);
__global__ void CUDA_zoom_in(int maxitemcount, Vec3f* rawarray);
__global__ void CUDA_zoom_out(int maxitemcount, Vec3f* rawarray);
void PEGA_start_fps_benchmark(void);
void PEGA_get_fps_benchmark(void);

void PEGA_init_3D(void)
{
 enable_lighting = true;
 counter_vert = counter_vert = 0;
 MAX_3d_vertex_count = 0;
 zoom_value = 1.0;
 render_inprogress = 0;
 persp_deg = Math_PI / 180;
 rot_deg.x = 0 * Math_PI / 180; rot_deg2.x = 0;
 rot_deg.y = 0 * Math_PI / 180; rot_deg2.y = 0;
 rot_deg.z = 0 * Math_PI / 180; rot_deg2.z = 0;
 raw_vertices_length = 0;
 vLight.x = -0.5;
 vLight.y = -0.5;
 vLight.z = -0.9;
}

void PEGA_3D_vertex_reset(void)
{
 raw_vertices_length = 0;
}

void PEGA_create_HOST_CUDA_3D_vertex_list(int count)
{
 int i;
 MAX_3d_vertex_count = count;
 raw_vertices_length = 0;
 raw_vertices = (Vec3f*)malloc(MAX_3d_vertex_count * sizeof(Vec3f));

 cudaSetDevice(0);
 cudaMalloc((void**)&CUDA_DEV0_COPY_Zbuffer, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(float));

 for (i = 0; i < CUDA_DEVICE_COUNT; ++i)
 {
  cudaSetDevice(i);
  cudaMalloc((void**)&CUDA_raw_vertices[i], (MAX_3d_vertex_count / CUDA_DEVICE_COUNT) * sizeof(Vec3f));
  cudaMalloc((void**)&CUDA_rotated_vertices[i], (MAX_3d_vertex_count / CUDA_DEVICE_COUNT) * sizeof(Vec3f));
  cudaMalloc((void**)&CUDA_Zbuffer[i], SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(float));
 }
}

void PEGA_free3D(void)
{
 int i;
 free(raw_vertices);
 cudaSetDevice(0);
 cudaFree(CUDA_DEV0_COPY_Zbuffer);
 for (i = 0; i < CUDA_DEVICE_COUNT; ++i)
 {
  cudaSetDevice(i);
  cudaFree(CUDA_raw_vertices[i]);
  cudaFree(CUDA_rotated_vertices[i]);
  cudaFree(CUDA_Zbuffer[i]);
 }
}

void PEGA_rotate_3D(void)
{
 int i;
 int blockSize = CUDA_CORES;
 int numBlocks;
 Vec3f deg_sin, deg_cos;

 rot_deg.x = rot_deg2.x * Math_PI / 180;
 rot_deg.y = rot_deg2.y * Math_PI / 180;
 rot_deg.z = rot_deg2.z * Math_PI / 180;
 deg_sin.x = sin(rot_deg.x);
 deg_cos.x = cos(rot_deg.x);
 deg_sin.y = sin(rot_deg.y);
 deg_cos.y = cos(rot_deg.y);
 deg_sin.z = sin(rot_deg.z);
 deg_cos.z = cos(rot_deg.z);

 for (i = 0; i < CUDA_DEVICE_COUNT; ++i)
 {
  if (PEGA_DEVICES.devinfo[i].work_items[PEGA3DTRIANGLE] > 0)
  {
   numBlocks = ((PEGA_DEVICES.devinfo[i].work_items[PEGA3DTRIANGLE] * 3) + blockSize - 1) / blockSize;
   cudaSetDevice(i);
   PEGA_3D_rotation << <numBlocks, blockSize >> > (PEGA_DEVICES.devinfo[i].work_items[PEGA3DTRIANGLE] * 3, CUDA_raw_vertices[i], CUDA_rotated_vertices[i], deg_cos, deg_sin);
  }
 }
 cudaDeviceSynchronize();
}

void PEGA_render_3D(void)
{
 int i;
 int blockSize = CUDA_CORES;
 int numBlocks;
 PEGA_clear_zbuffer();

 for (i = 0; i < CUDA_DEVICE_COUNT; ++i)
 {
  if (PEGA_DEVICES.devinfo[i].work_items[PEGA3DTRIANGLE] > 0)
  {
   numBlocks = ((PEGA_DEVICES.devinfo[i].work_items[PEGA3DTRIANGLE] * 3) + blockSize - 1) / blockSize;
   cudaSetDevice(i);
   //watch out for using 3 for other obejct types
   PEGA_render_objects << <numBlocks, blockSize >> > ((PEGA_DEVICES.devinfo[i].work_items[PEGA3DTRIANGLE] * 3), CUDA_rotated_vertices[i], CUDA_imagebuffer[i], CUDA_Zbuffer[i], vLight, enable_lighting);
  }
 }
 cudaDeviceSynchronize();
}

void PEGA_merge_down_zbuffers(void)
{
 int i;
 int blockSize = CUDA_CORES;
 int numBlocks;

 cudaSetDevice(0);
 if (CUDA_DEVICE_COUNT > 1)
 {
  for (i = 1; i < CUDA_DEVICE_COUNT; ++i)
  {
   if (PEGA_DEVICES.devinfo[i].work_items[PEGA3DTRIANGLE] > 0)
   {
    cudaMemcpyPeerAsync(CUDA_DEV0_COPY_Zbuffer, 0, CUDA_Zbuffer[i], i, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(float));
    cudaMemcpyPeerAsync(CUDA_DEV0_COPY_imagebuffer, 0, CUDA_imagebuffer[i], i, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(unsigned int));
    numBlocks = ((PEGA_DEVICES.devinfo[i].work_items[PEGA3DTRIANGLE] * 3) + blockSize - 1) / blockSize;    
    CUDA_Merge_Zbuffers << <numBlocks, blockSize >> > (CUDA_Zbuffer[0], CUDA_DEV0_COPY_Zbuffer, CUDA_imagebuffer[0], CUDA_DEV0_COPY_imagebuffer);
    //cudaDeviceSynchronize();
   }
  }
  cudaDeviceSynchronize();
 }
}

__global__ void CUDA_Merge_Zbuffers(float* main_zpuffer, float* secondary_zpuffer, unsigned int* main_imagedata, unsigned int* secondary_imagedata)
{
 int i;
 int index = (blockIdx.x * blockDim.x) + threadIdx.x;
 int stride = blockDim.x * gridDim.x;

 for (i = index; i < SCREEN_HEIGHT * SCREEN_WIDTH; i += stride)
 {
  if (main_zpuffer[i] > secondary_zpuffer[i])
  {
   main_imagedata[i] = secondary_imagedata[i];
  }
 }
}

__global__ void PEGA_render_objects(int maxitemcount, Vec3f* rotarray, unsigned int* puffer, float* zpuffer, Vec3f lightvektor, bool islightenabled)
{
 int i, customcolor;
 int index = (blockIdx.x * blockDim.x) + (threadIdx.x * 3);
 int stride = blockDim.x * gridDim.x;
 Vec3f Vector1, Vector2, vNormal;//for visibility check
 Vec3f vNormalized;//for lighting
 float Light_intensity, Vector_length;

 for (i = index; i < maxitemcount - 3; i += stride)
 {
  if ((rotarray[i].z < -9000000) || (rotarray[i + 1].z < -9000000) || (rotarray[i + 2].z < -9000000)) continue;

  // visibility check
  Vector1.x = rotarray[i + 1].x - rotarray[i].x;
  Vector1.y = rotarray[i + 1].y - rotarray[i].y;
  Vector1.z = rotarray[i + 1].z - rotarray[i].z;
  Vector2.x = rotarray[i + 2].x - rotarray[i].x;
  Vector2.y = rotarray[i + 2].y - rotarray[i].y;
  Vector2.z = rotarray[i + 2].z - rotarray[i].z;
  vNormal.x = ((Vector1.y * Vector2.z) - (Vector1.z * Vector2.y));
  vNormal.y = ((Vector1.z * Vector2.x) - (Vector1.x * Vector2.z));
  vNormal.z = ((Vector1.x * Vector2.y) - (Vector1.y * Vector2.x));
  if (vNormal.z > 0) continue;
  if (islightenabled == true)
  {
   Vector_length = sqrtf((vNormal.x * vNormal.x) + (vNormal.y * vNormal.y) + (vNormal.z * vNormal.z));
   vNormalized.x = vNormal.x / Vector_length;
   vNormalized.y = vNormal.y / Vector_length;
   vNormalized.z = vNormal.z / Vector_length;

   Light_intensity = ((vNormalized.x * lightvektor.x) + (vNormalized.y * lightvektor.y) + (vNormalized.z * lightvektor.z));
   if (Light_intensity > 1) Light_intensity = 1;
   else if (Light_intensity < 0) Light_intensity = 0;
   customcolor = RGB(255 * Light_intensity, 255 * Light_intensity, 255 * Light_intensity);
  }
  else customcolor = RGB(200, 0, 111);
  /*CUDA_SetPixel_Zbuffer(rotarray[i].x, rotarray[i].y, rotarray[i].z, customcolor, puffer, zpuffer);
  CUDA_SetPixel_Zbuffer(rotarray[i + 1].x, rotarray[i + 1].y, rotarray[i + 1].z, customcolor, puffer, zpuffer);
  CUDA_SetPixel_Zbuffer(rotarray[i + 2].x, rotarray[i + 2].y, rotarray[i + 2].z, customcolor, puffer, zpuffer);*/
  /*CUDA_DrawLine_Zbuffer(rotarray[i].x, rotarray[i].y, rotarray[i].z, rotarray[i + 1].x, rotarray[i + 1].y, rotarray[i + 1].z, customcolor, puffer, zpuffer);
  CUDA_DrawLine_Zbuffer(rotarray[i + 2].x, rotarray[i + 2].y, rotarray[i + 2].z, rotarray[i + 1].x, rotarray[i + 1].y, rotarray[i + 1].z, customcolor, puffer, zpuffer);
  CUDA_DrawLine_Zbuffer(rotarray[i].x, rotarray[i].y, rotarray[i].z, rotarray[i + 2].x, rotarray[i + 2].y, rotarray[i + 2].z, customcolor, puffer, zpuffer);*/
  CUDA_FillTriangle_Zbuffer(rotarray[i].x, rotarray[i].y, rotarray[i].z, rotarray[i + 1].x, rotarray[i + 1].y, rotarray[i + 1].z, rotarray[i + 2].x, rotarray[i + 2].y, rotarray[i + 2].z, customcolor, puffer, zpuffer);

 }
}

__global__ void PEGA_3D_rotation(int maxitemcount, Vec3f* rawarray, Vec3f* rotarray, Vec3f deg_cos, Vec3f deg_sin)
{
 int i;
 int index = blockIdx.x * blockDim.x + threadIdx.x;
 int stride = blockDim.x * gridDim.x;
 float t0;

 for (i = index; i < maxitemcount; i += stride)
 {
  rotarray[i].y = (rawarray[i].y * deg_cos.x) - (rawarray[i].z * deg_sin.x);
  rotarray[i].z = rawarray[i].y * deg_sin.x + rawarray[i].z * deg_cos.x;

  rotarray[i].x = rawarray[i].x * deg_cos.y + rotarray[i].z * deg_sin.y;
  rotarray[i].z = -rawarray[i].x * deg_sin.y + rotarray[i].z * deg_cos.y;

  t0 = rotarray[i].x;
  rotarray[i].x = t0 * deg_cos.z - rotarray[i].y * deg_sin.z;
  rotarray[i].y = t0 * deg_sin.z + rotarray[i].y * deg_cos.z;
 }

 //perspective calculations
 int viewpoint = -1100;
 float sx = SCREEN_WIDTH / 2;
 float sultra = SCREEN_HEIGHT / 2, sultra2 = SCREEN_HEIGHT / 3;
 int x_minus_limit = 0, y_minus_limit = 0, x_max_limit = SCREEN_WIDTH - 1, y_max_limit = SCREEN_HEIGHT - 1;
 float distance;

 for (i = index; i < maxitemcount; i += stride)
 {
  distance = 999999;

  if (rotarray[i].z < distance) distance = rotarray[i].z;
  if (distance < viewpoint) { rotarray[i].z = -9999999; continue; }//not displayable
  sultra = viewpoint / (viewpoint - rotarray[i].z);
  rotarray[i].x = rotarray[i].x * sultra + 400 + 400;
  rotarray[i].y = (rotarray[i].y * sultra) + sultra2 + 200;
  if (rotarray[i].x < x_minus_limit || rotarray[i].x > x_max_limit) { rotarray[i].z = -9999999; continue; }// not displayable
  if (rotarray[i].y < y_minus_limit || rotarray[i].y > y_max_limit) { rotarray[i].z = -9999999; continue; }// not displayable
 }
}

void PEGA_add_3D_triangle(int x1, int y1, int z1, int x2, int y2, int z2, int x3, int y3, int z3)
{
 if (raw_vertices_length + 3 >= MAX_3d_vertex_count) return;
 raw_vertices[raw_vertices_length].x = x1;
 raw_vertices[raw_vertices_length].y = y1;
 raw_vertices[raw_vertices_length++].z = z1;
 raw_vertices[raw_vertices_length].x = x2;
 raw_vertices[raw_vertices_length].y = y2;
 raw_vertices[raw_vertices_length++].z = z2;
 raw_vertices[raw_vertices_length].x = x3;
 raw_vertices[raw_vertices_length].y = y3;
 raw_vertices[raw_vertices_length++].z = z3;
}

void PEGA_push_3D_triangles_to_GPU(void)
{
 int i;
 if (raw_vertices_length > 0) {
  PEGA_calculate_load_distribution(raw_vertices_length / 3, PEGA3DTRIANGLE);//watch out for other object types
  for (i = 0; i < CUDA_DEVICE_COUNT; ++i)
  {
   if (PEGA_DEVICES.devinfo[i].work_items[PEGA3DTRIANGLE] > 0)
   {
    cudaSetDevice(i);
    //wacth out for 3 for other obejct types !
    cudaMemcpy(CUDA_raw_vertices[i], raw_vertices + (3 * PEGA_DEVICES.devinfo[i].start_item[PEGA3DTRIANGLE]), (3 * PEGA_DEVICES.devinfo[i].work_items[PEGA3DTRIANGLE]) * sizeof(Vec3f), cudaMemcpyHostToDevice);
   }
  }
 }
}

void PEGA_clear_zbuffer(void)
{
 int i;
 for (i = 0; i < CUDA_DEVICE_COUNT; ++i)
 {
  cudaSetDevice(i);
  PEGA_cleanup_zbuffer << < ((SCREEN_WIDTH * SCREEN_HEIGHT) + CUDA_CORES - 1) / CUDA_CORES, CUDA_CORES >> > (CUDA_Zbuffer[i]);
 }
 cudaDeviceSynchronize();
}

__global__ void PEGA_cleanup_zbuffer(float* zpuffer)
{
 int i;
 int index = (blockIdx.x * blockDim.x) + threadIdx.x;
 int stride = blockDim.x * gridDim.x;

 for (i = index; i < SCREEN_HEIGHT * SCREEN_WIDTH; i += stride)
 {
  zpuffer[i] = 999999;
 }
}

__device__ void CUDA_SetPixel_Zbuffer(int x1, int y1, int z1, int mycolor, unsigned int* puffer, float* zpuffer)
{
 int aktoffset = (y1 * SCREEN_WIDTH) + x1;
 if (zpuffer[aktoffset] > z1)
 {
  zpuffer[aktoffset] = z1;
  puffer[aktoffset] = mycolor;
 }
}

__device__ void CUDA_DrawLine_Zbuffer(int x1, int y1, int z1, int x2, int y2, int z2, int mycolor, unsigned int* puffer, float* zpuffer)
{
 float Pz;
 bool flip = false;
 int swapping, aktoffset;

 if (abs(x2 - x1) < 2 && abs(y2 - y1) < 2) {
  puffer[(y2 * SCREEN_WIDTH) + x2] = mycolor; return;
 }
 if (abs(x1 - x2) < abs(y1 - y2))
 {
  swapping = x1;
  x1 = y1;
  y1 = swapping;

  swapping = x2;
  x2 = y2;
  y2 = swapping;
  flip = true;
 }
 if (x1 > x2)
 {
  swapping = x1;
  x1 = x2;
  x2 = swapping;

  swapping = y1;
  y1 = y2;
  y2 = swapping;
 }
 int dx = x2 - x1;
 int dy = y2 - y1;

 int marker1 = abs(dy) * 2;
 int marker2 = 0;
 int y = y1, x;

 for (x = x1; x <= x2; ++x)
 {
  if (z1 == z2) Pz = z1;
  else
  {
   int s1 = abs(x2 - x1);
   int s2 = abs(z1 - z2);
   Pz = (float)z2 + (float)((((float)x - (float)x1) / (float)s1) * (float)s2);
  }
  if (flip)
  {
   aktoffset = (x * SCREEN_WIDTH);
   if (zpuffer[aktoffset + y] > Pz)
   {
    zpuffer[aktoffset + y] = Pz;
    puffer[aktoffset + y] = mycolor;
   }
  }
  else
  {
   aktoffset = (y * SCREEN_WIDTH);
   if (zpuffer[aktoffset + x] > Pz)
   {
    zpuffer[aktoffset + x] = Pz;
    puffer[aktoffset + x] = mycolor;
   }
  }
  marker2 += marker1;
  if (marker2 > dx)
  {
   y += (y2 > y1 ? 1 : -1);
   marker2 -= dx * 2;
  }
 }
}

__device__ void CUDA_FillTriangle_Zbuffer(int x1, int y1, int z1, int x2, int y2, int z2, int x3, int y3, int z3, int mycolor, unsigned int* puffer, float* zpuffer)
{
 int Ax, Ay, Bx, By, i, j, deepvalue;
 int swappingx, swappingy, aktoffset;
 Vec3f interpolal, helpervektor;
 if (y1 == y2 && y1 == y3) return;

 if (y1 > y2)
 {
  swappingx = x1;
  swappingy = y1;
  x1 = x2;
  y1 = y2;
  x2 = swappingx;
  y2 = swappingy;
 }
 if (y1 > y3)
 {
  swappingx = x1;
  swappingy = y1;
  x1 = x3;
  y1 = y3;
  x3 = swappingx;
  y3 = swappingy;
 }
 if (y2 > y3)
 {
  swappingx = x3;
  swappingy = y3;
  x3 = x2;
  y3 = y2;
  x2 = swappingx;
  y2 = swappingy;
 }
 int height = y3 - y1;
 for (i = 0; i < height; ++i)
 {
  bool second_half = i > y2 - y1 || y2 == y1;
  int segment_height = second_half ? y3 - y2 : y2 - y1;
  float alpha = (float)i / height;
  float beta = (float)(i - (second_half ? y2 - y1 : 0)) / segment_height;
  Ax = x1 + (x3 - x1) * alpha;
  Ay = y1 + (y3 - y1) * alpha;
  Bx = second_half ? x2 + (x3 - x2) * beta : x1 + (x2 - x1) * beta;
  By = second_half ? y2 + (y3 - y2) * beta : y1 + (y2 - y1) * beta;
  if (Ax > Bx)
  {
   swappingx = Ax;
   swappingy = Ay;
   Ax = Bx;
   Ay = By;
   Bx = swappingx;
   By = swappingy;
  }

  aktoffset = (y1 + i) * SCREEN_WIDTH;
  for (j = Ax; j <= Bx; ++j)
  {
   helpervektor.x = (x2 - x1) * (y1 - (y1 + i)) - (x1 - j) * (y2 - y1);
   helpervektor.y = (x1 - j) * (y3 - y1) - (x3 - x1) * (y1 - (y1 + i));
   helpervektor.z = (x3 - x1) * (y2 - y1) - (x2 - x1) * (y3 - y1);
   if (abs((int)helpervektor.z) < 1) { interpolal.x = -1; interpolal.y = 0; interpolal.z = 0; }
   else
   {
    interpolal.x = 1.f - (helpervektor.x + helpervektor.y) / helpervektor.z;
    interpolal.y = helpervektor.y / helpervektor.z;
    interpolal.z = helpervektor.x / helpervektor.z;
   }
   if (interpolal.x < 0 || interpolal.y < 0 || interpolal.z < 0) continue;
   deepvalue = (z1 * interpolal.x) + (z2 * interpolal.y) + (z3 * interpolal.z);
   if (zpuffer[aktoffset + j] > deepvalue)
   {
    zpuffer[aktoffset + j] = deepvalue;
    puffer[aktoffset + j] = mycolor;
   }
  }
 }
}

void PEGA_zoom_in(void)
{
 int i;
 int blockSize = CUDA_CORES;
 int numBlocks;
 for (i = 0; i < CUDA_DEVICE_COUNT; ++i)
 {
  if (PEGA_DEVICES.devinfo[i].work_items[PEGA3DTRIANGLE] > 0)
  {
   numBlocks = ((PEGA_DEVICES.devinfo[i].work_items[PEGA3DTRIANGLE] * 3) + blockSize - 1) / blockSize;
   cudaSetDevice(i);
   CUDA_zoom_in << <numBlocks, blockSize >> > (PEGA_DEVICES.devinfo[i].work_items[PEGA3DTRIANGLE] * 3, CUDA_raw_vertices[i]);
  }
 }
 cudaDeviceSynchronize();
}

void PEGA_zoom_out(void)
{
 int i;
 int blockSize = CUDA_CORES;
 int numBlocks;
 for (i = 0; i < CUDA_DEVICE_COUNT; ++i)
 {
  if (PEGA_DEVICES.devinfo[i].work_items[PEGA3DTRIANGLE] > 0)
  {
   numBlocks = ((PEGA_DEVICES.devinfo[i].work_items[PEGA3DTRIANGLE] * 3) + blockSize - 1) / blockSize;
   cudaSetDevice(i);
   CUDA_zoom_out << <numBlocks, blockSize >> > (PEGA_DEVICES.devinfo[i].work_items[PEGA3DTRIANGLE] * 3, CUDA_raw_vertices[i]);
  }
 }
 cudaDeviceSynchronize();
}

__global__ void CUDA_zoom_in(int maxitemcount, Vec3f* rawarray)
{
 int i;
 int index = (blockIdx.x * blockDim.x) + threadIdx.x;
 int stride = blockDim.x * gridDim.x;
 for (i = index; i < maxitemcount; i += stride)
 {
  rawarray[i].x *= 1.2;
  rawarray[i].y *= 1.2;
  rawarray[i].z *= 1.2;
 }
}

__global__ void CUDA_zoom_out(int maxitemcount, Vec3f* rawarray)
{
 int i;
 int index = (blockIdx.x * blockDim.x) + threadIdx.x;
 int stride = blockDim.x * gridDim.x;
 for (i = index; i < maxitemcount; i += stride)
 {
  rawarray[i].x /= 1.2;
  rawarray[i].y /= 1.2;
  rawarray[i].z /= 1.2;
 }
}

void PEGA_start_fps_benchmark(void)
{
 bmark_beginning = GetTickCount();
}

void PEGA_get_fps_benchmark(void)
{
 bmark_ending = GetTickCount();
 if ((bmark_ending - bmark_beginning) == 0) ++bmark_ending;
 fps_stat = 1000 / (bmark_ending - bmark_beginning);
}
