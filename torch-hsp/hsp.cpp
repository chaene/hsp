#include <stdio.h>
#include <math.h>
#include <malloc.h>
//#include <png.h>
#include <luaT.h>
#include <TH/TH.h>

#include "CImg/CImg.h"
#include <iostream>
#include <fstream>

#define INF 1e20

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

/* 
 * Parts of this file are copied form the public code of:
 * 
 * Distance Transforms of Sampled Functions
 * P. Felzenszwalb, D. Huttenlocher
 * Theory of Computing, Vol. 8, No. 19, September 2012 
 * 
 * https://cs.brown.edu/~pff/dt/index.html
 * 
 */

extern "C" {

double square(const double x) { return x*x; };

struct Volume
{
  double* data;
  int width;
  int height;
  int depth;
};

void makeVolume(struct Volume *vol, int width, int height, int depth)
{
  vol->data = (double*) malloc(sizeof(double)*width*height*depth);
  vol->width = width;
  vol->height = height;
  vol->depth = depth;
};

size_t voxelIndex(struct Volume* vol, int x, int y, int z)
{
  return (size_t)z*(vol->height*vol->width) + y*vol->width + x;
};

void sqrtVolume(struct Volume *vol)
{
  for (int z=0; z < vol->depth; ++z)
  {
    for(int y=0; y < vol->height; ++y)
    {
      for(int x=0; x < vol->width; ++x)
      {
        vol->data[voxelIndex(vol, x, y, z)] = sqrt(vol->data[voxelIndex(vol, x, y, z)]);
      }
    }
  }
}


/* dt of 1d function using squared distance */
static double *dt(double *f, int n) {
  double *d = (double*) malloc(sizeof(double)*n);
  int *v = (int*) malloc(sizeof(int)*n);
  double *z = (double*) malloc(sizeof(double)*(n+1));
  int k = 0;
  v[0] = 0;
  z[0] = -INF;
  z[1] = +INF;
  for (int q = 1; q <= n-1; q++) {
    float s  = ((f[q]+square(q))-(f[v[k]]+square(v[k])))/(2*q-2*v[k]);
    while (s <= z[k]) {
      k--;
      s  = ((f[q]+square(q))-(f[v[k]]+square(v[k])))/(2*q-2*v[k]);
    }
    k++;
    v[k] = q;
    z[k] = s;
    z[k+1] = +INF;
  }

  k = 0;
  for (int q = 0; q <= n-1; q++) {
    while (z[k+1] < q)
      k++;
    d[q] = square(q-v[k]) + f[v[k]];
  }

  free(v);
  free(z);
  return d;
  
};


static void dt3D(struct Volume *vol) {
  const int width = vol->width;
  const int height = vol->height;
  const int depth = vol->depth;
  double *f = (double*) malloc(sizeof(double)*MAX(MAX(width,height),depth));
  
  // transform along planes

  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      for (int z = 0; z < depth; z++)
      {
        f[z] = vol->data[voxelIndex(vol, x, y, z)];
      }
      
      double *d = dt(f, depth);
      
      for (int z = 0; z < depth; z++)
      {
        vol->data[voxelIndex(vol, x, y, z)] = d[z];
      }
      free(d);
    }
  }

  // transform along columns
  for (int z = 0; z < depth; z++)
  {
    for (int x = 0; x < width; x++)
    {
      for (int y = 0; y < height; y++)
      {
        f[y] = vol->data[voxelIndex(vol, x, y, z)];
      }
      
      double *d = dt(f, height);
      
      for (int y = 0; y < height; y++)
      {
        vol->data[voxelIndex(vol, x, y, z)] = d[y];
      }
      free(d);
    }
  }

  // transform along rows
  
  for (int z = 0; z < depth; z++)
  {
    for (int y = 0; y < height; y++)
    {
      for (int x = 0; x < width; x++)
      {
        f[x] =  vol->data[voxelIndex(vol, x, y, z)];
      }
      
      double *d = dt(f, width);
      
      for (int x = 0; x < width; x++)
      {
        vol->data[voxelIndex(vol, x, y, z)] = d[x];
      }
      free(d);
    }
  }
  free(f);
}

static struct Volume extractBoundary(struct Volume *vol, const double onValue, const double bVal, const double nonBVal)
{
  const int width = vol->width;
  const int height = vol->height;
  const int depth = vol->depth;
  
  struct Volume bVol;
  makeVolume(&bVol, width, height, depth);
  
  
  for (int z = 0; z < depth; z++)
  {
    for (int y = 0; y < height; y++)
    {
      for (int x = 0; x < width; x++)
      {
	int boundary = 0;
	
	if (vol->data[voxelIndex(vol, x, y, z)] == onValue)
	{
	  for (int zz = MAX(z-1,0); zz <= MIN(z+1,depth-1) && !boundary; ++zz)
	    for (int yy = MAX(y-1,0); yy <= MIN(y+1,height-1) && !boundary; ++yy)
	      for (int xx = MAX(x-1,0); xx <= MIN(x+1,width-1) && !boundary; ++xx)
		if (vol->data[voxelIndex(vol, xx, yy, zz)] != onValue)
		  boundary = 1;
	}
	
	if (boundary)
	  bVol.data[voxelIndex(&bVol, x, y, z)] = bVal;
	else
	  bVol.data[voxelIndex(&bVol, x, y, z)] = nonBVal;
      }
    }
  }
  
  return bVol;
}


static int boundary(lua_State *L)
{
  int w,h,d;
  THDoubleTensor *tensor = (THDoubleTensor *)luaT_checkudata(L, 1, "torch.DoubleTensor");
  h = THDoubleTensor_size(tensor,0);
  w = THDoubleTensor_size(tensor,1);
  d = THDoubleTensor_size(tensor,2);
  
  double* tensorData = (double*)THDoubleTensor_data(tensor);
  struct Volume inputVol;
  
  inputVol.width=w;
  inputVol.height=h;
  inputVol.depth=d;
  inputVol.data = tensorData;
  
  struct Volume bVol = extractBoundary(&inputVol, 1, 1, 0);
  
  memcpy(inputVol.data, bVol.data, sizeof(double)*h*w*d);
  
  //TODO: make a nicer function for this
  free(bVol.data);
  
  return 1;
}


static int boundaryEDT(lua_State *L)
{
  int w,h,d;
  THDoubleTensor *tensor = (THDoubleTensor *)luaT_checkudata(L, 1, "torch.DoubleTensor");
  h = THDoubleTensor_size(tensor,0);
  w = THDoubleTensor_size(tensor,1);
  d = THDoubleTensor_size(tensor,2);

  double* tensorData = (double*)THDoubleTensor_data(tensor);
  struct Volume inputVol;
  
  inputVol.width=w;
  inputVol.height=h;
  inputVol.depth=d;
  inputVol.data = tensorData;
  
  struct Volume bVol = extractBoundary(&inputVol, 1, 0, INF);
  

  dt3D(&bVol);
  sqrtVolume(&bVol);
  
  memcpy(inputVol.data, bVol.data, sizeof(double)*h*w*d);
  
  //TODO: make a nicer function for this
  free(bVol.data);
  
  return 1;
  
}

static int saveMeshAsObj(lua_State *L)
{
  
  int w,h,d;
  THDoubleTensor *volume = (THDoubleTensor *)luaT_checkudata(L, 1, "torch.DoubleTensor");
  double threshold = lua_tonumber(L, 2);
  w = THDoubleTensor_size(volume, 0);
  h = THDoubleTensor_size(volume, 1);
  d = THDoubleTensor_size(volume, 2);
  
  double* volumeData = (double*)THDoubleTensor_data(volume);
  
  // create cimg volume
  cimg_library::CImg<double> volCImg(w, h, d, 1);
  
  for (unsigned int x=0; x < w; ++x) {
    for (unsigned int y=0; y < h; ++y) {
      for (unsigned int z=0; z < d; ++z) {
        volCImg(x,y,z) = volumeData[z*(h*w) + y*w + x];
      }
    }
  }
  
  // run the CImg MC implementation
  cimg_library::CImgList<unsigned int> faces3d;
  const cimg_library::CImg<float> points3d = volCImg.get_isosurface3d(faces3d,threshold);
  
  // writing the file
  const char* outputFileName = lua_tostring(L, 3);
  
  std::ofstream stream;
  stream.open(outputFileName);
  
  if (!stream.is_open()) {
    std::cerr << "Could not open output file: " << outputFileName << std::endl;
    return -1;
  }
  
  const float deltaX = 1.0/w;
  const float deltaY = 1.0/h;
  const float deltaZ = 1.0/d;
  
  for (int i=0; i < points3d.width(); ++i) {
    float pointX = -0.5 + deltaX*0.5 + deltaX*points3d(i,0);
    float pointY = -0.5 + deltaY*0.5 + deltaY*points3d(i,1);
    float pointZ = -0.5 + deltaZ*0.5 + deltaZ*points3d(i,2);
    stream << "v " << pointX << " " << pointY << " " << pointZ << std::endl; 
  }
  
  for (unsigned int i=0; i < faces3d.size(); ++i) {
    stream << "f ";
    for (int j = faces3d(i).height()-1; j >= 0; --j) {
       stream << faces3d(i)(0,j)+1 << " ";
    }
    stream << std::endl;
  }
}

static int saveColoredMeshAsPLY(lua_State *L)
{
  
  int w,h,d;
  THDoubleTensor *volume = (THDoubleTensor *)luaT_checkudata(L, 1, "torch.DoubleTensor");
  THDoubleTensor *colorVolume = (THDoubleTensor *)luaT_checkudata(L, 2, "torch.DoubleTensor");
  double threshold = lua_tonumber(L, 3);
  w = THDoubleTensor_size(volume, 0);
  h = THDoubleTensor_size(volume, 1);
  d = THDoubleTensor_size(volume, 2);
  
  double* volumeData = (double*)THDoubleTensor_data(volume);
  double* colorData = (double*)THDoubleTensor_data(colorVolume);
  
  // create cimg volume
  cimg_library::CImg<double> volCImg(w, h, d, 1);
  
  for (unsigned int x=0; x < w; ++x) {
    for (unsigned int y=0; y < h; ++y) {
      for (unsigned int z=0; z < d; ++z) {
        volCImg(x,y,z) = volumeData[z*(h*w) + y*w + x];
      }
    }
  }
  
  // run the CImg MC implementation
  cimg_library::CImgList<unsigned int> faces3d;
  const cimg_library::CImg<float> points3d = volCImg.get_isosurface3d(faces3d,threshold);
  
  // writing the file
  const char* outputFileName = lua_tostring(L, 4);
  
  std::ofstream stream;
  stream.open(outputFileName);
  
  if (!stream.is_open()) {
    std::cerr << "Could not open output file: " << outputFileName << std::endl;
    return -1;
  }
  
  // write header
  stream << "ply" << std::endl;
  stream << "format ascii 1.0" << std::endl;
  stream << "element vertex " << points3d.width() << std::endl;
  stream << "property float x" << std::endl;
  stream << "property float y" << std::endl;
  stream << "property float z" << std::endl;
  stream << "property uchar red" << std::endl;
  stream << "property uchar green" << std::endl;
  stream << "property uchar blue" << std::endl;
  stream << "element face " << faces3d.size() << std::endl;
  stream << "property list uchar int vertex_index" << std::endl; 
  stream << "end_header" << std::endl;
  
  const float deltaX = 1.0/w;
  const float deltaY = 1.0/h;
  const float deltaZ = 1.0/d;
  
  for (int i=0; i < points3d.width(); ++i) {
    float pointX = -0.5 + deltaX*0.5 + deltaX*points3d(i,0);
    float pointY = -0.5 + deltaY*0.5 + deltaY*points3d(i,1);
    float pointZ = -0.5 + deltaZ*0.5 + deltaZ*points3d(i,2);
    
    float colIndX = points3d(i,0);
    float colIndY = points3d(i,1);
    float colIndZ = points3d(i,2);
    
    // trilinear interpolation
    
    int colIndX0 = std::floor(colIndX);
    int colIndY0 = std::floor(colIndY);
    int colIndZ0 = std::floor(colIndZ);
    
    int colIndX1 = colIndX0 < w-1 ? colIndX0 + 1 : colIndX0;
    int colIndY1 = colIndY0 < h-1 ? colIndY0 + 1 : colIndY0;
    int colIndZ1 = colIndZ0 < d-1 ? colIndZ0 + 1 : colIndZ0;
    
    float xd = (colIndX - colIndX0);
    float yd = (colIndY - colIndY0);
    float zd = (colIndZ - colIndZ0);
    
    float rc00 = colorData[h*w*d*0 + colIndZ0*(h*w) + colIndY0*w + colIndX0]*(1-xd) + colorData[h*w*d*0 + colIndZ0*(h*w) + colIndY0*w + colIndX1]*xd;
    float gc00 = colorData[h*w*d*1 + colIndZ0*(h*w) + colIndY0*w + colIndX0]*(1-xd) + colorData[h*w*d*1 + colIndZ0*(h*w) + colIndY0*w + colIndX1]*xd;
    float bc00 = colorData[h*w*d*2 + colIndZ0*(h*w) + colIndY0*w + colIndX0]*(1-xd) + colorData[h*w*d*2 + colIndZ0*(h*w) + colIndY0*w + colIndX1]*xd;
    float rc01 = colorData[h*w*d*0 + colIndZ1*(h*w) + colIndY0*w + colIndX0]*(1-xd) + colorData[h*w*d*0 + colIndZ1*(h*w) + colIndY0*w + colIndX1]*xd;
    float gc01 = colorData[h*w*d*1 + colIndZ1*(h*w) + colIndY0*w + colIndX0]*(1-xd) + colorData[h*w*d*1 + colIndZ1*(h*w) + colIndY0*w + colIndX1]*xd;
    float bc01 = colorData[h*w*d*2 + colIndZ1*(h*w) + colIndY0*w + colIndX0]*(1-xd) + colorData[h*w*d*2 + colIndZ1*(h*w) + colIndY0*w + colIndX1]*xd;
    float rc10 = colorData[h*w*d*0 + colIndZ0*(h*w) + colIndY1*w + colIndX0]*(1-xd) + colorData[h*w*d*0 + colIndZ0*(h*w) + colIndY1*w + colIndX1]*xd;
    float gc10 = colorData[h*w*d*1 + colIndZ0*(h*w) + colIndY1*w + colIndX0]*(1-xd) + colorData[h*w*d*1 + colIndZ0*(h*w) + colIndY1*w + colIndX1]*xd;
    float bc10 = colorData[h*w*d*2 + colIndZ0*(h*w) + colIndY1*w + colIndX0]*(1-xd) + colorData[h*w*d*2 + colIndZ0*(h*w) + colIndY1*w + colIndX1]*xd;
    float rc11 = colorData[h*w*d*0 + colIndZ1*(h*w) + colIndY1*w + colIndX0]*(1-xd) + colorData[h*w*d*0 + colIndZ1*(h*w) + colIndY1*w + colIndX1]*xd;
    float gc11 = colorData[h*w*d*1 + colIndZ1*(h*w) + colIndY1*w + colIndX0]*(1-xd) + colorData[h*w*d*1 + colIndZ1*(h*w) + colIndY1*w + colIndX1]*xd;
    float bc11 = colorData[h*w*d*2 + colIndZ1*(h*w) + colIndY1*w + colIndX0]*(1-xd) + colorData[h*w*d*2 + colIndZ1*(h*w) + colIndY1*w + colIndX1]*xd;
    
    float rc0 = rc00*(1-yd) + rc10*yd;
    float gc0 = gc00*(1-yd) + gc10*yd;
    float bc0 = bc00*(1-yd) + bc10*yd;
    float rc1 = rc01*(1-yd) + rc11*yd;
    float gc1 = gc01*(1-yd) + gc11*yd;
    float bc1 = bc01*(1-yd) + bc11*yd;
    
    int colR = std::floor((rc0*(1-zd)+rc1*zd)*255);
    int colG = std::floor((gc0*(1-zd)+gc1*zd)*255);
    int colB = std::floor((bc0*(1-zd)+bc1*zd)*255);
    
    //int colR = std::floor(colorData[h*w*d*0 + colIndZ*(h*w) + colIndY*w + colIndX]*255);
    //int colG = std::floor(colorData[h*w*d*1 + colIndZ*(h*w) + colIndY*w + colIndX]*255);
    //int colB = std::floor(colorData[h*w*d*2 + colIndZ*(h*w) + colIndY*w + colIndX]*255);
    
    // transform point such that it gets loaded properly in blender
    stream << pointX << " " << -pointZ << " " << pointY << " " << colR << " " << colG << " " << colB << std::endl; 
  }
  
  for (unsigned int i=0; i < faces3d.size(); ++i) {
    stream << "3 ";
    for (int j = faces3d(i).height()-1; j >= 0; --j) {
       stream << faces3d(i)(0,j) << " ";
    }
    stream << std::endl;
  }
}

static int addDepthMapToColorVolume(lua_State *L)
{
  int w,h,volRes;
  THDoubleTensor *maskVolume = (THDoubleTensor *)luaT_checkudata(L, 1, "torch.DoubleTensor");
  THDoubleTensor *colorVolume = (THDoubleTensor *)luaT_checkudata(L, 2, "torch.DoubleTensor");
  THDoubleTensor *depthMap = (THDoubleTensor *)luaT_checkudata(L, 3, "torch.DoubleTensor");
  THDoubleTensor *colorImage = (THDoubleTensor *)luaT_checkudata(L, 4, "torch.DoubleTensor");
  THDoubleTensor *Kinv = (THDoubleTensor *)luaT_checkudata(L, 5, "torch.DoubleTensor");
  THDoubleTensor *R = (THDoubleTensor *)luaT_checkudata(L, 6, "torch.DoubleTensor");
  THDoubleTensor *C = (THDoubleTensor *)luaT_checkudata(L, 7, "torch.DoubleTensor");
  volRes = THDoubleTensor_size(colorVolume, 1);
  w = THDoubleTensor_size(depthMap, 1);
  h = THDoubleTensor_size(depthMap, 2);
  
  double* maskVolumeData = (double*)THDoubleTensor_data(maskVolume);
  double* colorVolumeData = (double*)THDoubleTensor_data(colorVolume);
  double* depthMapData = (double*)THDoubleTensor_data(depthMap);
  double* colorImageData = (double*)THDoubleTensor_data(colorImage);
  double* KinvData = (double*)THDoubleTensor_data(Kinv);
  double* RData = (double*)THDoubleTensor_data(R);
  double* CData = (double*)THDoubleTensor_data(C);
  
  //std::cout << Kinv->stride[1] << std::endl;
  
  int KinvS0 = Kinv->stride[0];
  int KinvS1 = Kinv->stride[1];
  
  int RS0 = R->stride[0];
  int RS1 = R->stride[1];
  
  int dmS1 = depthMap->stride[1];
  int dmS2 = depthMap->stride[2];
  
  int cIS0 = colorImage->stride[0];
  int cIS1 = colorImage->stride[1];
  int cIS2 = colorImage->stride[2];
  
  int mS1 = maskVolume->stride[1];
  int mS2 = maskVolume->stride[2];
  int mS3 = maskVolume->stride[3];
  
  int cS0 = colorVolume->stride[0];
  int cS1 = colorVolume->stride[1];
  int cS2 = colorVolume->stride[2];
  int cS3 = colorVolume->stride[3];
  

  for (int y = 0; y < h; ++y)
  {
    for (int x = 0; x < h; ++x)
    {
      double pointImgX = x;
      double pointImgY = y;
      
      double pointLocalX = (pointImgX*KinvData[0] + pointImgY*KinvData[KinvS1*1] + KinvData[KinvS1*2])*depthMapData[y*dmS1 + x*dmS2];
      double pointLocalY = (pointImgX*KinvData[KinvS0*1 + KinvS1*0] + pointImgY*KinvData[KinvS0*1 + KinvS1*1] + KinvData[KinvS0*1 + KinvS1*2])*depthMapData[y*dmS1 + x*dmS2]; 
      double pointLocalZ = (KinvData[KinvS0*2 + KinvS1*0] + KinvData[KinvS0*2 + KinvS1*1] + KinvData[KinvS0*2 + KinvS1*2])*depthMapData[y*dmS1 + x*dmS2];
      
      double pointGlobalBlendX = RData[RS0*0 + RS1*0]*pointLocalX + RData[RS0*0 + RS1*1]*pointLocalY + RData[RS0*0 + RS1*2]*pointLocalZ + CData[0];
      double pointGlobalBlendY = RData[RS0*1 + RS1*0]*pointLocalX + RData[RS0*1 + RS1*1]*pointLocalY + RData[RS0*1 + RS1*2]*pointLocalZ + CData[1];
      double pointGlobalBlendZ = RData[RS0*2 + RS1*0]*pointLocalX + RData[RS0*2 + RS1*1]*pointLocalY + RData[RS0*2 + RS1*2]*pointLocalZ + CData[2];
      
      double pointGlobalX = pointGlobalBlendX;
      double pointGlobalY = pointGlobalBlendZ;
      double pointGlobalZ = -pointGlobalBlendY;
      
      int indexX = floor((pointGlobalX + 0.5)*volRes);
      int indexY = floor((pointGlobalY + 0.5)*volRes);
      int indexZ = floor((pointGlobalZ + 0.5)*volRes);
      
      if (indexX >= 0 && indexY >= 0 && indexZ >= 0 && indexX < volRes && indexY < volRes && indexZ < volRes)
      {
        maskVolumeData[indexX*mS1 + indexY*mS2 + indexZ*mS3] += 1;
        colorVolumeData[cS0*0 + indexX*cS1 + indexY*cS2 + indexZ*cS3] += colorImageData[cIS0*0 + cIS1*y + cIS2*x];
        colorVolumeData[cS0*1 + indexX*cS1 + indexY*cS2 + indexZ*cS3] += colorImageData[cIS0*1 + cIS1*y + cIS2*x];
        colorVolumeData[cS0*2 + indexX*cS1 + indexY*cS2 + indexZ*cS3] += colorImageData[cIS0*2 + cIS1*y + cIS2*x];
      }
    }
  }
  
  return 1;
}


static const luaL_Reg hsplib[] = {
  {"saveMeshAsObj", saveMeshAsObj},
  {"saveColoredMeshAsPLY", saveColoredMeshAsPLY},
  {"boundaryEDT", boundaryEDT},
  {"boundary", boundary},
  {"addDepthMapToColorVolume", addDepthMapToColorVolume},
  {NULL, NULL}
};

int luaopen_libhsp (lua_State *L) {
  luaL_openlib(L, "libhsp", hsplib, 0);
  return 1;
}

}