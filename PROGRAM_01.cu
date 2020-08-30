#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <time.h> 
#include <windows.h>
#include <d2d1.h>
#include <d2d1helper.h>
#pragma comment(lib, "d2d1")
#include <mmsystem.h>
#pragma comment(lib, "winmm.lib")

#include "pegazus_main.h"
#include "PEGA_FORMAT_OBJ_NEW.h"
#include "PEGA_FORMAT_BMP.h"

//***********STANDARD WIN32API WINDOWS HANDLING************
HINSTANCE hInstGlob;
int SajatiCmdShow;
HWND Form1; //Windows handler
LRESULT CALLBACK WndProc0(HWND, UINT, WPARAM, LPARAM);
//******************************************************

//*********************************
//Entry point of the program
//*********************************
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PSTR szCmdLine, int iCmdShow)
{
 MSG msg;
 WNDCLASS wndclass0;
 SajatiCmdShow = iCmdShow;
 hInstGlob = hInstance;

 //*********************************
 //Preparing the Window class instance
 //*********************************
 wndclass0.style = CS_HREDRAW | CS_VREDRAW;
 wndclass0.lpfnWndProc = WndProc0;
 wndclass0.cbClsExtra = 0;
 wndclass0.cbWndExtra = 0;
 wndclass0.hInstance = hInstance;
 wndclass0.hIcon = LoadIcon(NULL, IDI_APPLICATION);
 wndclass0.hCursor = LoadCursor(NULL, IDC_ARROW);
 wndclass0.hbrBackground = (HBRUSH)GetStockObject(LTGRAY_BRUSH);
 wndclass0.lpszMenuName = NULL;
 wndclass0.lpszClassName = TEXT("WIN0");

 //*********************************
 //Registering the window class
 //*********************************
 if (!RegisterClass(&wndclass0))
 {
  MessageBox(NULL, TEXT("Error:Program initialisation process."), TEXT("Program Start"), MB_ICONERROR);
  return 0;
 }

 //*********************************
 //Creating the window
 //*********************************
 Form1 = CreateWindow(TEXT("WIN0"),
  TEXT("CUDA – Multi-GPU"),
  (WS_OVERLAPPED | WS_SYSMENU | WS_THICKFRAME | WS_MAXIMIZEBOX | WS_MINIMIZEBOX),
  0,
  0,
  SCREEN_WIDTH,
  SCREEN_HEIGHT,
  NULL,
  NULL,
  hInstance,
  NULL);

 //*********************************
 //Showing the window
 //*********************************
 ShowWindow(Form1, SajatiCmdShow);
 UpdateWindow(Form1);

 //*********************************
 //Window message handling
 //*********************************
 while (GetMessage(&msg, NULL, 0, 0))
 {
  TranslateMessage(&msg);
  DispatchMessage(&msg);
 }
 return msg.wParam;
}

//*********************************
//The window’s callback function: event handling
//*********************************
LRESULT CALLBACK WndProc0(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
 HDC hdc;
 PAINTSTRUCT ps;
 FILE* myfile;
 unsigned int xPos, yPos, xPos2, yPos2, fwButtons;

 switch (message)
 {
 //*********************************
 //Creating the window
 //*********************************
 case WM_CREATE:
  /*Init*/;  
  srand((unsigned)time(NULL));
  return 0;
 //*********************************
 //To prevent flickering
 //*********************************
 case WM_ERASEBKGND:
  return (LRESULT)1;

 //*********************************
 //Drawing the window’s client area
 //*********************************
 case WM_PAINT:
  hdc = BeginPaint(hwnd, &ps);
  EndPaint(hwnd, &ps);
  return 0;
 //*********************************
 //Closing the window
 //*********************************
 case WM_CLOSE:
  DestroyWindow(hwnd);
  return 0;
 //*********************************
 //Destroying the window
 //*********************************
 case WM_DESTROY:
  PostQuitMessage(0);
  return 0;
 }
 return DefWindowProc(hwnd, message, wParam, lParam);
}
