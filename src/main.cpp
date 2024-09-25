
#include <iostream>
#include <string.h>
#include<unistd.h>
#include<sys/types.h>
#include <errno.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>
typedef int ShareableHandle;
#define CHECK_CUDA_ERROR(call)                               \
    {                                                        \
        cudaError_t err = (call);                            \
        if (err != cudaSuccess) {                            \
            std::cerr << "CUDA error in " << __FILE__        \
                      << " at line " << __LINE__ << ": "     \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                              \
        }                                                    \
    }
#define ROUND_UP(x,n) ((x + (n - 1)) & (~(n - 1)))

const size_t buf_size = 1920*1080*3*10;
void parent_process(int fd){
    CUcontext ctx;
    CUdevice device;
    CUresult result = cuInit(0);
    cuDeviceGet(&device, 0); // 获取第0个设备
    cuCtxCreate(&ctx, 0, device);

    int deviceSupportsVmm = 0;
    result = cuDeviceGetAttribute(&deviceSupportsVmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device);
    if (result != CUDA_SUCCESS) {
        printf("Failed to get device attribute, error code :%d \n",result);
    }
    //分配内存
    
    size_t granularity = 0;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0; //在device 0上分配内存
    prop.requestedHandleTypes  = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR; //允许句柄被导出
    cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    size_t padded_size = ROUND_UP(buf_size, granularity);
    CUmemGenericAllocationHandle allocHandle;
    std::cout<<"allocation minimum: "<< granularity <<"\tneed size: "<<buf_size <<"\tpadding size: "<<padded_size<<std::endl;
    cuMemCreate(&allocHandle, padded_size, &prop, 0);
    //查询平台是否支持内存导出
    int deviceSupportsIpcHandle;
#if defined(__linux__)
    cuDeviceGetAttribute(&deviceSupportsIpcHandle, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, device);
#else
    cuDeviceGetAttribute(&deviceSupportsIpcHandle, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED, device);
#endif
    std::cout<<"is support mem export: "<<deviceSupportsIpcHandle<<std::endl;
    CUdeviceptr ptr;

    result = cuMemAddressReserve(&ptr, padded_size, 0, 0, 0); // alignment = 0 for default alignment
    std::cout<<"cuMemAddressReserve: "<<(result==CUDA_SUCCESS)<<std::endl;
    result = cuMemMap(ptr, padded_size, 0, allocHandle, 0);
    std::cout<<"cuMemMap: "<<((result==CUDA_SUCCESS)?1:result) <<std::endl;


    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = device;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    cuMemSetAccess(ptr, padded_size, &accessDesc, 1);

    constexpr int num =20;
    int *host_data1 = new int[num];
    for(int i =0;i<num;i++ ){
        if(i == 0){
            host_data1[i] = std::numeric_limits<int>::max();
        }else{
            host_data1[i] = i;
        }
    }
    //cuMemHostRegister(ptr, padded_size, CU_MEMHOSTREGISTER_PORTABLE);
    result = cuMemcpyHtoD(ptr,(void *)host_data1, sizeof(int)*num);
    std::cout<<"cuMemcpyHtoD,success = "<<((result==CUDA_SUCCESS)?1:result)<<"\tresult "<<result <<std::endl;

    //导出句柄
    ShareableHandle shareable_handle;
    result = cuMemExportToShareableHandle((void *)&shareable_handle,allocHandle,CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR , 0);
    std::cout<<"1 1 "<<result <<std::endl;
    //发送句柄
    ssize_t nbyte = write(fd,(void *)&shareable_handle ,sizeof(ShareableHandle));	
    std::cout<<"write byte: "<< nbyte<<" shared handle: "<<shareable_handle<<std::endl;
    close(fd);
    int *host_data2 = new int[num];
    memset(host_data2,0,num*sizeof(int));
    while (1)
    { 
        result =  cuMemcpyDtoH((void *)host_data2,ptr,sizeof(int)*num);
        if(host_data2 && host_data2[0] == 0){
            break;
        }
    }

    delete host_data1;
    delete host_data2;
    cuMemUnmap(ptr,padded_size); 
    cuMemAddressFree(ptr,padded_size);
    cuMemRelease(allocHandle);
    cuCtxDestroy(ctx);
    std::cout<<"parent process exit....";
}
void child_process(int fd){
    int shareable_handle;
    int nbyte =0;
    std::cout << "begin child process.... "<< std::endl;
    do{
        nbyte = read(fd, (void *)&shareable_handle, sizeof(int));
        if(nbyte > 0 ){
            std::cout << "read byte: "<<nbyte <<"\tshared handle: " << shareable_handle  << std::endl;
        }
    }while(nbyte<=0);
    close(fd);


    CUcontext ctx;
    CUdevice device;
    CUresult result = cuInit(0);
    std::cout<<"0 "<<result<<std::endl;
    cuDeviceGet(&device, 0); // 获取第0个设备
    cuCtxCreate(&ctx, 0, device);
    const int map_size = 62914560;
    size_t granularity = 0;

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0; //在device 0上分配内存
    prop.requestedHandleTypes  = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR; //允许句柄被导出
    cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

   
    //cuMemCreate(&allocHandle, map_size, &prop, 0);
  
    CUdeviceptr d_ptr;
    result = cuMemAddressReserve(&d_ptr, map_size, 0, 0, 0); // alignment = 0 for default alignment 
    std::cout<<"1 "<<result << std::endl;     
    
    CUmemGenericAllocationHandle allocHandle;               
    
    result = cuMemImportFromShareableHandle(&allocHandle, (void *)&shareable_handle,CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    std::cout<<"2 |"<<result<<"|" << std::endl;       
    result = cuMemMap(d_ptr, map_size, 0, allocHandle, 0);
    std::cout<<"3 "<<result << std::endl; 

    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = device;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    result = cuMemSetAccess(d_ptr,map_size,&accessDesc, 1);
    std::cout<<"4 "<<result << std::endl; 

    const int num =20;
    int *host_data = new int[num];
    result =  cuMemcpyDtoH((void *)host_data,d_ptr,sizeof(int)*num);
    std::cout<<"5 "<<result << std::endl; 
    std::cout<<"child read data...."<<std::endl;
    for(int i = 0;i<num;i++){
        std::cout<<" i = "<<host_data[i]<<" ";
    }
    std::cout<<std::endl;
    host_data[0] = std::numeric_limits<int>::max();
    result = cuMemcpyHtoD(d_ptr,(void *)host_data, sizeof(int)*1);
    std::cout<<"6 "<<result << std::endl; 
    delete host_data;
    cuMemUnmap(d_ptr,map_size); 
    cuMemAddressFree(d_ptr,map_size);
    cuMemRelease(allocHandle);
    cuCtxDestroy(ctx);
    std::cout<<"child process exit....";
}
int main(int argc,char** argv){
    
	//创建管道
	int fd[2];
	int ret = pipe(fd);
	if(ret<0)
	{
		perror("pipe error");
		return -1;
    }
    pid_t pid = fork();
    if(pid == 0){
        close(fd[1]);
        child_process(fd[0]);
        exit(0);
    }else{
        //关闭读端
		close(fd[0]);
        parent_process(fd[1]);
    } 
    return 0;   
}