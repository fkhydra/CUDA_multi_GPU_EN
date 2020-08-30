#pragma once
#define SCREEN_WIDTH 2500
#define SCREEN_HEIGHT 1400
#define CUDA_CORES 384
#define CUDA_BLOCKS 24
#define CUDA_MAX_DEVICES 4
#define PEGA_EVEN_DISTRIBUTION 1
#define PEGA_SEQUENTIAL_DISTRIBUTION 2

int PEGA_load_distribution_mode;
bool PEGA_isdebuggingenabled = true;
unsigned int* CUDA_imagebuffer[CUDA_MAX_DEVICES], *CUDA_DEV0_COPY_imagebuffer;
unsigned int* CUDA_maskimagebuffer[CUDA_MAX_DEVICES];//0-empty,1-valid

int CUDA_DEVICE_COUNT;

typedef struct Vec3f {
 float x, y, z;
};

typedef struct PEGA_SHAPE_2D_POINT {
 int x1, y1, color;
};

typedef struct PEGA_SHAPE_2D_LINE {
 int x1, y1, x2, y2, color;
};

typedef struct PEGA_SHAPE_2D_TRIANGLE {
 int x1, y1, x2, y2, x3, y3, color;
};

typedef struct PEGA_CUDA_DEVICE {
 int cores, memsize, work_items[4], start_item[4];
};
#define PEGA2DPOINT 0
#define PEGA2DLINE 1
#define PEGA2DTRIANGLE 2
#define PEGA3DTRIANGLE 3

typedef struct PEGA_DEVICES_DEF {
 PEGA_CUDA_DEVICE devinfo[CUDA_MAX_DEVICES];
 int devcount;
};
PEGA_DEVICES_DEF PEGA_DEVICES;

int PEGA_benchmark_beginning, PEGA_benchmark_end, PEGA_benchmark_duration;

void PEGA_set_load_distribution_mode(int modenum);
void PEGA_limit_device_count(int devcount);
void PEGA_benchmark_start(void);
void PEGA_benchmark_stop(char* title);
void PEGA_calculate_load_distribution(int itemcount, int PEGATYPE);
int get_rnd(int maxnum);
void PEGA_log_dev_status(int rawvertexcount, int max3dvertexcount);

void PEGA_set_load_distribution_mode(int modenum)
{
 PEGA_load_distribution_mode = modenum;
}

void PEGA_limit_device_count(int devcount)
{
 CUDA_DEVICE_COUNT = devcount;
}

void PEGA_benchmark_start(void)
{
 if (PEGA_isdebuggingenabled == true) PEGA_benchmark_beginning = GetTickCount();
}

void PEGA_benchmark_stop(char *title)
{
 if (PEGA_isdebuggingenabled == false) return;
 FILE* myfile;
 PEGA_benchmark_end = GetTickCount();
 PEGA_benchmark_duration = PEGA_benchmark_end - PEGA_benchmark_beginning;
 fopen_s(&myfile, "CUDA_benchmark.txt", "at");
 if (myfile != NULL)
 {
  fprintf_s(myfile, "%s: %d\n", title, PEGA_benchmark_duration);
 }
 fclose(myfile);
}

void PEGA_calculate_load_distribution(int itemcount, int PEGATYPE)
{
 int i, j, k, temp = 0, temp2, bastardcount = 0;

 if (PEGA_load_distribution_mode == PEGA_EVEN_DISTRIBUTION)
 {
  for (i = temp2 = 0; i < CUDA_DEVICE_COUNT; ++i) temp2 += PEGA_DEVICES.devinfo[i].cores;
  
  if (itemcount < temp2) //1 device is enough
  {
   for (i = 0; i < CUDA_DEVICE_COUNT; ++i)
   {
    PEGA_DEVICES.devinfo[i].work_items[PEGATYPE] = 0;
   }
   PEGA_DEVICES.devinfo[0].work_items[PEGATYPE] = itemcount;
   //calculating bastard elements
   temp2 = 0;
   bastardcount = 0;// itemcount - temp2;
  }
  else
  {
   temp = itemcount / temp2;
   for (i = 0; i < CUDA_DEVICE_COUNT; ++i)
   {
    PEGA_DEVICES.devinfo[i].work_items[PEGATYPE] = temp * PEGA_DEVICES.devinfo[i].cores;
   }
   // calculating bastard elements
   for (i = temp2 = 0; i < CUDA_DEVICE_COUNT; ++i)
   {
    temp2 += (temp * PEGA_DEVICES.devinfo[i].cores);
   }
   bastardcount = itemcount - temp2;
   PEGA_DEVICES.devinfo[CUDA_DEVICE_COUNT - 1].work_items[PEGATYPE] += bastardcount;
  }

  PEGA_DEVICES.devinfo[0].start_item[PEGATYPE] = 0;
  for (i = 1; i < CUDA_DEVICE_COUNT; ++i)
  {
   for (j = temp2 = 0; j <= i - 1; ++j)
   {
    temp2 += PEGA_DEVICES.devinfo[j].work_items[PEGATYPE];
   }
   PEGA_DEVICES.devinfo[i].start_item[PEGATYPE] = temp2;
  }
 }
 else if (PEGA_load_distribution_mode == PEGA_SEQUENTIAL_DISTRIBUTION)
 {
  unsigned long long tempavailablememory, temp_workitems;
  switch (PEGATYPE)
  {
  case PEGA2DPOINT:temp = sizeof(PEGA_SHAPE_2D_POINT);
   break;
  case PEGA2DLINE:temp = sizeof(PEGA_SHAPE_2D_LINE);
   break;
  case PEGA2DTRIANGLE:temp = sizeof(PEGA_SHAPE_2D_TRIANGLE);
   break;
  case PEGA3DTRIANGLE:temp = sizeof(Vec3f) * 3;
   break;
  default:temp = 1;
   break;
  }
  temp_workitems = itemcount * temp;

  for (i = 0; i < CUDA_DEVICE_COUNT; ++i)
  {
   PEGA_DEVICES.devinfo[i].work_items[PEGATYPE] = 0;
   PEGA_DEVICES.devinfo[i].start_item[PEGATYPE] = 0;
  }
  
  for (i = 0; i < CUDA_DEVICE_COUNT; ++i)
  {
   tempavailablememory = (PEGA_DEVICES.devinfo[i].memsize * 1024 * 1024 * 1024) - (400 * 1024 * 1024);
   if (temp_workitems <= tempavailablememory)
   {
    PEGA_DEVICES.devinfo[i].work_items[PEGATYPE] = temp_workitems / temp;
    break;
   }
   else
   {
    PEGA_DEVICES.devinfo[i].work_items[PEGATYPE] = tempavailablememory / temp;
    temp_workitems -= (tempavailablememory / temp);
   }
  }  
 } 

 if (PEGA_isdebuggingenabled == true)
 {
  int temp3;
  FILE* myfile;
  fopen_s(&myfile, "LOG_DEV_distribution.txt", "wt");
  if (myfile != NULL)
  {
   fprintf_s(myfile, "Data items to be processed: %d\n", itemcount);
   fprintf_s(myfile, "----------------------\n");
   fprintf_s(myfile, "Bastard elements: %d\n", bastardcount);
   fprintf_s(myfile, "----------------------\n");
   for (i = temp3 = 0; i < CUDA_DEVICE_COUNT; ++i)
   {
    if (PEGA_DEVICES.devinfo[i].work_items[PEGATYPE] > 0)
     temp3 += PEGA_DEVICES.devinfo[i].cores;
   }    
   fprintf_s(myfile, "%d on each core\n", (itemcount / temp3));
   for (i = 0; i < CUDA_DEVICE_COUNT; ++i)
   {
    if (PEGA_DEVICES.devinfo[i].work_items[PEGATYPE] > 0) fprintf_s(myfile, "DEV%d: %d\n", i + 1, PEGA_DEVICES.devinfo[i].work_items[PEGATYPE]);
   }
   fprintf_s(myfile, "----------------------\n");
   fprintf_s(myfile, "Starting item indices\n");
   
   fprintf_s(myfile, "DEV1: 0\n");//first item
   for (i = 1; i < CUDA_DEVICE_COUNT; ++i)
   {
    for (j = temp2 = 0; j <= i - 1; ++j)
    {
     temp2 += PEGA_DEVICES.devinfo[j].work_items[PEGATYPE];
    }
    if(PEGA_DEVICES.devinfo[j].work_items[PEGATYPE] > 0) fprintf_s(myfile, "DEV%d: %d\n", i + 1, temp2);
   }
  }
  fclose(myfile);
 } 
}

int get_rnd(int maxnum)
{
 return (double)rand() / (RAND_MAX + 1) * (maxnum - 0);
}

void PEGA_log_dev_status(int rawvertexcount,int max3dvertexcount)
{
 if (PEGA_isdebuggingenabled == false) return;
 FILE* myfile;
 fopen_s(&myfile, "LOG_DEV_status.txt", "wt");
 if (myfile != NULL)
 {
  fprintf_s(myfile, "%d vertices (%d MB)\n", rawvertexcount, rawvertexcount * sizeof(Vec3f) / 1024 / 1024);
  fprintf_s(myfile, "%d MB vertex cache allocated on all devices\n", (max3dvertexcount / CUDA_DEVICE_COUNT) * sizeof(Vec3f) / 1024 / 1024);
  int i;
  for (i = 0; i < CUDA_DEVICE_COUNT; ++i)
  {
   fprintf_s(myfile, "%d. device\n", i);
   fprintf_s(myfile, "Number of CUDA cores: %d\n", PEGA_DEVICES.devinfo[i].cores);
   fprintf_s(myfile, "Memory size(GB): %d\n", PEGA_DEVICES.devinfo[i].memsize);
  }
 }
 fclose(myfile);
}
