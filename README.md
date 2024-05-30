# FastSAM_rknn_Cplusplus
FastSAM 部署rknn C++ 代码


## 编译和运行

1）编译

```
cd examples/rknn_fastSAM

bash build-linux_RK3588.sh

```

2）运行

```
cd install/rknn_FastSAM_demo_Linux

./rknn_FastSAM_demo

```

注意：修改模型、测试图像、保存图像的路径，修改文件为src下的main.cc

```

int main(int argc, char **argv)
{
    char model_path[256] = "/home/firefly/zhangqian/rknn/examples/FastSAM_rknn_Cplusplusmodel/RK3588/FastSAM-x_640x352.rknn";
    char image_path[256] = "/home/firefly/zhangqian/rknn/examples/FastSAM_rknn_Cplusplus/test.jpg";
    char save_image_path[256] = "/home/firefly/zhangqian/rknn/examples/FastSAM_rknn_Cplusplus/test_result.jpg";

    detect(model_path, image_path, save_image_path);
    return 0;
}
```


# 测试效果

![image](https://github.com/cqu20160901/FastSAM_rknn_Cplusplus/blob/main/examples/rknn_fastSAM/test_result.jpg)

把板端模型推理和后处理时耗也附上，供参考，使用的芯片rk3588，模型输入640x352。
![image](https://github.com/cqu20160901/FastSAM_rknn_Cplusplus/assets/22290931/cbe59747-085a-4f88-bdd7-a6388026071f)




