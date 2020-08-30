#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "pegazus_config.h"
#pragma once

PEGA_SHAPE_2D_POINT* points2D;
int points2D_counter;
int points2D_MAX;

PEGA_SHAPE_2D_LINE* lines2D;
int lines2D_counter;
int lines2D_MAX;

PEGA_SHAPE_2D_TRIANGLE* triangles2D;
int triangles2D_counter;
int triangles2D_MAX;

PEGA_SHAPE_2D_LINE* CUDA_lines2D[CUDA_MAX_DEVICES];
PEGA_SHAPE_2D_POINT* CUDA_points2D[CUDA_MAX_DEVICES];
PEGA_SHAPE_2D_TRIANGLE* CUDA_triangles2D[CUDA_MAX_DEVICES];

__device__ void CUDA_SetPixel2D(int x1, int y1, int mycolor, unsigned int* puffer, unsigned int* maskpuffer);
__device__ void CUDA_DrawLine2D(int x1, int y1, int x2, int y2, int mycolor, unsigned int* puffer, unsigned int* maskpuffer);
__device__ void CUDA_FillTriangle(int x1, int y1, int x2, int y2, int x3, int y3, int mycolor, unsigned int* puffer, unsigned int* maskpuffer);

__device__ void CUDA_SetPixel2D(int x1, int y1, int mycolor, unsigned int* puffer, unsigned int* maskpuffer)
{
 int i;
 puffer[(y1 * SCREEN_WIDTH) + x1] = mycolor;
 maskpuffer[(y1 * SCREEN_WIDTH) + x1] = 1;
}

__device__ void CUDA_DrawLine2D(int x1, int y1, int x2, int y2, int mycolor, unsigned int* puffer, unsigned int* maskpuffer)
{
 bool flip = false;
 int swapping, aktoffset;

 if (abs(x2 - x1) < 2 && abs(y2 - y1) < 2)
 {
  aktoffset = (y2 * SCREEN_WIDTH);
  puffer[aktoffset + x2] = mycolor;
  maskpuffer[aktoffset + x2] = 1;
  return;
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

 if (flip)
 {
  for (x = x1; x <= x2; ++x)
  {
   aktoffset = (x * SCREEN_WIDTH);
   puffer[aktoffset + y] = mycolor;
   maskpuffer[aktoffset + y] = 1;
   marker2 += marker1;
   if (marker2 > dx)
   {
    y += (y2 > y1 ? 1 : -1);
    marker2 -= dx * 2;
   }
  }
 }
 else
 {
  for (x = x1; x <= x2; ++x)
  {
   aktoffset = (y * SCREEN_WIDTH);
   puffer[aktoffset + x] = mycolor;
   maskpuffer[aktoffset + y] = 1;
   marker2 += marker1;
   if (marker2 > dx)
   {
    y += (y2 > y1 ? 1 : -1);
    marker2 -= dx * 2;
   }
  }
 }
}

__device__ void CUDA_FillTriangle(int x1, int y1, int x2, int y2, int x3, int y3, int mycolor, unsigned int* puffer, unsigned int* maskpuffer)
{
 int Ax, Ay, Bx, By, i, j;
 int swappingx, swappingy, aktoffset, maxoffset = SCREEN_HEIGHT * SCREEN_WIDTH;
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
  bool lowerpart = i > y2 - y1 || y2 == y1;
  int partialheight = lowerpart ? y3 - y2 : y2 - y1;
  float alpha = (float)i / height;
  float beta = (float)(i - (lowerpart ? y2 - y1 : 0)) / partialheight;
  Ax = x1 + (x3 - x1) * alpha;
  Ay = y1 + (y3 - y1) * alpha;
  Bx = lowerpart ? x2 + (x3 - x2) * beta : x1 + (x2 - x1) * beta;
  By = lowerpart ? y2 + (y3 - y2) * beta : y1 + (y2 - y1) * beta;
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
  for (j = Ax; j < Bx; ++j)
  {
   if (aktoffset + j > maxoffset) continue;
   puffer[aktoffset + j] = mycolor;
   maskpuffer[aktoffset + j] = 1;
  }
 }
}
