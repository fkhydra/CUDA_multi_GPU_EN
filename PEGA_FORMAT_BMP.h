#pragma once
#include <stdio.h>
#include <d2d1.h>

void PEGA_load_BMP(const char* filename, unsigned int* picbuffer);

void PEGA_load_BMP(const char *filename, unsigned int* picbuffer)
{
 FILE* bmpfile;
 int i, j, s = 0;
 unsigned char R, G, B;
 bmpfile = fopen(filename, "rb");
 if (bmpfile == NULL) return;
 fseek(bmpfile, 54, SEEK_SET);
 for (i = 0; i < 1080; ++i)
  for (j = 0; j < 1920; ++j)
  {
   fread(&B, 1, 1, bmpfile);
   fread(&G, 1, 1, bmpfile);
   fread(&R, 1, 1, bmpfile);
   picbuffer[s++] = RGB(B, G, R);
  }
 fclose(bmpfile);
}
