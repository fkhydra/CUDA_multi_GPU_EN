#pragma once
#pragma once
#include "pegazus_config.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#define MAX_OBJ_NUM 80000000

float* array_vertices0, * array_vertices1, * array_vertices2;
int array_vertices_length;

int PEGA_getelementcount(unsigned char csv_mytext[]);
void PEGA_getelement(unsigned char csv_mytext[], unsigned int whichisit, unsigned char csv_mytext2[]);
void PEGA_obj_loader(void);

void PEGA_allocate_OBJ_data(void)
{
 array_vertices0 = (float*)malloc(MAX_OBJ_NUM * sizeof(float));
 array_vertices1 = (float*)malloc(MAX_OBJ_NUM * sizeof(float));
 array_vertices2 = (float*)malloc(MAX_OBJ_NUM * sizeof(float));
}

void PEGA_free_OBJ_data(void)
{
 free(array_vertices0);
 free(array_vertices1);
 free(array_vertices2);
}

int PEGA_getelementcount(unsigned char csv_mytext[])
{
 int s1, s2;
 for (s1 = s2 = 0; s1 < strlen((const char*)csv_mytext); ++s1)
 {
  if (csv_mytext[s1] == 10) break;
  else if (csv_mytext[s1] == 32) ++s2;
 }
 return s2;
}

void PEGA_getelement(unsigned char csv_mytext[], unsigned int whichisit, unsigned char csv_mytext2[])
{
 int s1, s2, s3, s4 = 0;
 for (s1 = 0, s2 = 0; s1 < strlen((const char*)csv_mytext); ++s1)
 {
  if (csv_mytext[s1] == 32)
  {
   ++s2;
   if (s2 == whichisit)
   {
    for (s3 = s1 + 1; s3 < strlen((const char*)csv_mytext); ++s3)
    {
     if (csv_mytext[s3] == 32 || csv_mytext[s3] == 10)
     {
      csv_mytext2[s4] = 0;
      return;
     }
     else csv_mytext2[s4++] = csv_mytext[s3];
    }
   }
  }
 }
}

void PEGA_obj_loader(char *filename)
{
 FILE* myfile;
 int i, j;
 float data1, data2, data3, data4;
 int idata1, idata2, idata3, idata4;
 unsigned char row1[1024], row2[1024];
 int itemcount, maxrowlength = 250;

 myfile = fopen("lion2.obj", "rt");
 if (myfile == NULL) return;

 PEGA_allocate_OBJ_data();
 array_vertices_length = 0;

 while (!feof(myfile))
 {
  fgets((char*)row1, maxrowlength, myfile);

  if (row1[0] == 118 && row1[1] == 32) //*** 'v '
  {
   PEGA_getelement(row1, 1, row2); data1 = atof((const char*)row2);
   PEGA_getelement(row1, 2, row2); data2 = atof((const char*)row2);
   PEGA_getelement(row1, 3, row2); data3 = atof((const char*)row2);
   if (array_vertices_length >= MAX_OBJ_NUM) continue;
   array_vertices0[array_vertices_length] = data1 * 1000;
   array_vertices1[array_vertices_length] = data2 * 1000;
   array_vertices2[array_vertices_length++] = data3 * 1000;
  }
  else if (row1[0] == 102 && row1[1] == 32) //*** 'f '
  {
   itemcount = PEGA_getelementcount(row1);
   PEGA_getelement(row1, 1, row2);//3 vertices at least assumed
   idata1 = atoi((const char*)row2) - 1;
   PEGA_getelement(row1, 2, row2);
   idata2 = atoi((const char*)row2) - 1;
   PEGA_getelement(row1, 3, row2);
   idata3 = atoi((const char*)row2) - 1;
   PEGA_add_3D_triangle(array_vertices0[idata1], array_vertices1[idata1], array_vertices2[idata1],
    array_vertices0[idata2], array_vertices1[idata2], array_vertices2[idata2],
    array_vertices0[idata3], array_vertices1[idata3], array_vertices2[idata3]);

   if (itemcount == 4)
   {
    PEGA_getelement(row1, 4, row2);
    idata4 = atoi((const char*)row2) - 1;
    PEGA_add_3D_triangle(array_vertices0[idata1], array_vertices1[idata1], array_vertices2[idata1],
     array_vertices0[idata2], array_vertices1[idata2], array_vertices2[idata2],
     array_vertices0[idata4], array_vertices1[idata4], array_vertices2[idata4]);
   }
  }
 }
 fclose(myfile);
 PEGA_free_OBJ_data();
}
