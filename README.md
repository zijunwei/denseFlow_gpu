## Usage
This repo is for using GPU to extract videos' dense optical flow fields.

## Authorship
The code is mainly borrowed from [Limin Wang's repo](https://github.com/wanglimin/dense_flow).  
I've slightly changed the interface to facilitate my own usage.  


## Interface
Option | Name | Default | Note 
:---   | :--- | :---    | :---
f  | vidFile    | ex.avi  | filename of video
x  | xFlowFile  | x       | filename prefix of flow x component
y  | yFlowFile  | y       | filename prefix of flow x component
i  | imgFile    | i       | filename prefix of image
b  | bound      | 20      | specify the maximum (px) of optical flow
t  | type       | 1       | specify the optical flow algorithm
d  | device_id  | 0       | specify gpu id
s  | step       | 1       | specify the step for frame sampling
h  | height     | 0       | specify the height of saved flows, 0: keep original height
w  | width      | 0       | specify the width of saved flows,  0: keep original width

## Example
```
mkdir flow
./denseFlow_gpu -f ex.avi -x flow/x -y flow/y -i flow/i -b 20 -t 1 -d 0 -s 1 -h 0 -w 0
```
> ex.avi is the input video  
> flow/ is the folder containing the extracted RGB frames and optical flows

## Notes

* The RGB frames and optical flows are extracted at the original video resolution (Height, Width, FPS).  
* To extract optical flows at a different resolution, I recommend (1) reformat the video to that resolution, (2) extract frames/flows using this repo.
* Options -h & -w controls the resolution of stored frames/flows.


## Extending Yang's Work

1. Support on OpenCV 3 + 

2. Command when installing OpenCV 3 + 

```apple js
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_CUDA=ON -D WITH_CUBLAS=ON -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF -DCUDA_NVCC_FLAGS="-D_FORCE_INLINES" -D WITH_LIBV4L=ON  -D CUDA_GENERATION=Auto -D WITH_OPENMP=ON  ..

```

3. Changed Commandlineparse on OPENCV:

```apple js
./denseFlow_gpu -f=ex.avi -x=flow/x -y=flow/y -i=flow/i -b=20 -t=1 -d=0 -s=1 -h=0 -w=0

```