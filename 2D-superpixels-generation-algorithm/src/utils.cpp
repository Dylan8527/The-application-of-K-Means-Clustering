#include <utils.h>
#include <iostream>

void LoadARGBImage(
    	string&			filename,
	UINT*&				imgBuffer,
	int&				width,
	int&				height)
{
   Gdiplus::GdiplusStartupInput gdiplusStartupInput;
   ULONG_PTR           gdiplusToken;

   GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);
   printf("Loading image %s\n", filename.c_str());
   Bitmap* bmp				= 0;
   bmp = Bitmap::FromFile(string2wstring(filename).c_str());
   Status __retVal = bmp->GetLastStatus();
   if(__retVal != Status::Ok)
   {
       printf("Error loading image %s\n", filename.c_str());
       return;
   }
	height					= bmp->GetHeight();
	width					= bmp->GetWidth();
	printf("Image size: %d x %d\n", width, height);
	long imgSize			= height*width;
	Rect rect(0, 0, width, height);

	BitmapData*	bmpData		= new BitmapData;
	bmp->LockBits(
		&rect,
		ImageLockModeWrite,
		PixelFormat32bppARGB,
		bmpData);

	_ASSERT( bmpData->Stride/4 == width );

	if( bmpData->Stride/4 != width )
		return;//picture format may not be 24 bit jpg or bmp type

	imgBuffer = new UINT[imgSize];

	memcpy( imgBuffer, (UINT*)bmpData->Scan0, imgSize*sizeof(UINT) );

	bmp->UnlockBits(bmpData);
}

void SaveImage(
    UINT*&				imgBuffer,
	int					width,
	int					height,
	string&				savepath)
{
    int sz = width*height;
   printf("Saving image to %s\n", savepath.c_str());

	Bitmap bmp(width, height, width*sizeof(UINT), PixelFormat32bppARGB, (unsigned char *)imgBuffer);
    // Save bmp to savepath
    CLSID picClsid;
    GetEncoderClsid(L"image/jpeg", &picClsid);
    
   wstring wholepath = string2wstring(savepath);
	const WCHAR* wp = wholepath.c_str();

    Status st = bmp.Save( wp, &picClsid, NULL );
    _ASSERT( st == Ok );
}

int GetEncoderClsid(const WCHAR* format, CLSID* pClsid)
{
   UINT  num = 0;          // number of image encoders
   UINT  size = 0;         // size of the image encoder array in bytes

   ImageCodecInfo* pImageCodecInfo = NULL;

   GetImageEncodersSize(&num, &size);
   if(size == 0)
      return -1;  // Failure

   pImageCodecInfo = (ImageCodecInfo*)(malloc(size));
   if(pImageCodecInfo == NULL)
      return -1;  // Failure

   GetImageEncoders(num, size, pImageCodecInfo);

   for(UINT j = 0; j < num; ++j)
   {
      if( wcscmp(pImageCodecInfo[j].MimeType, format) == 0 )
      {
         *pClsid = pImageCodecInfo[j].Clsid;
         free(pImageCodecInfo);
         return j;  // Success
      }    
   }

   free(pImageCodecInfo);
   return -1;  // Failure
}

string wstring2string(const wstring& ws)
{
   _bstr_t t = ws.c_str();
   return string((char*)t);
}

wstring string2wstring(const string& s)
{
   _bstr_t t = s.c_str();
   return wstring((wchar_t*)t);
}