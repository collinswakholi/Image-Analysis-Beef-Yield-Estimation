from cx_Freeze import setup,Executable
 
setup(name = "deepPredict",
      version ="1.0",
      description = " when image (256 x 256 x 3) and hdf5 model is passed to it, produces binary image and blend of predicted features... by Collins Wakholi",
      executables = [Executable(r"F:\Collins_ops\Deep_learning\Deep_201906\Im_seg\read_predict_write.py")]
      )