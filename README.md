Camera Lens Super-Resolution
====
Chang Chen, Zhiwei Xiong, Xinmei Tian, Zheng-Jun Zha, Feng Wu. Camera Lens Super-Resolution. In CVPR 2019. <br/>

## City100 Dataset
To access the DSLR variant of City100
```
cd City100/City100_NikonD5500
```
To access the smartphone variant of City100
```
cd City100/City100_iPhoneX
```
To access the raw files of City100 (10 GB) <br/>
Download City100_RAW.tar.gz.0~3 from  <br/>
http://pan.bitahub.com/index.php?mod=shares&sid=eTJ2bFFQR3BzTm5FTGdWN19RNGh1TTR4d0Q3Y2dKT3NwNGVvSUE
```
cat City100_RAW.tar.gz.* | tar -xzv
```

## Test the pre-trained models
Usage example to VDSR model for reconstruction accuracy <br/>
```
cd Models/VDSR && python inference.py
```
Usage example to SRGAN model for perceptual quality <br/>
```
cd Models/SRGAN/model
cat model.data-00000-of-00001.tar.gz.* | tar -xzv
cd .. && /bin/bash inference.sh
```
