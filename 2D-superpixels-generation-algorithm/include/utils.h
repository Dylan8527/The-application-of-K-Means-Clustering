#ifndef _UTILS_H_
#define _UTILS_H_

#include "stdafx.h"
#include <gdiplus.h>
#include <vector>
#include <algorithm>
#include <string>
#include <mbctype.h>  
#include <comutil.h>
#pragma comment(lib, "comsuppw.lib")
#pragma comment(lib, "Gdiplus.lib")

namespace Gdiplus	{
					class  Bitmap;
					class  Graphics;
					struct GdiplusStartupInput;
					}

using namespace Gdiplus;
using namespace std;

typedef unsigned int UINT;

void LoadARGBImage(
    	string&			filename,
	UINT*&				imgBuffer,
	int&				width,
	int&				height);

void SaveImage(
    UINT*&				imgBuffer,
	int					width,
	int					height,
	string&				savepath);


int GetEncoderClsid(const WCHAR* format, CLSID* pClsid);
					string wstring2string(const wstring& ws);
					wstring string2wstring(const string& s);
#endif // _UTILS_H_