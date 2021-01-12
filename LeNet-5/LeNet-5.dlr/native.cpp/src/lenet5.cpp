//------------------------------------------------------------------------------
// Copyright (c) 2020 by Future Design Systems
// All right reserved.
//
// http://www.future-ds.com
//------------------------------------------------------------------------------
// VERSION = 2020.10.04.
//------------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cmath>

#if !defined(DTYPE)
#define DTYPE float
#endif

#include "dlr.hpp"
#include "lenet5_params.h"  // come from LeNet-5.pytorch

#define EMBED_ReLU   1

#ifndef EMBED_ReLU
#define EMBED_ReLU   1
#endif
// There may raise mis-match when EMBED_ReLU 0

#if defined(__SYNTHESIS__)
    #define PRAGMA_SUB(x) _Pragma (#x)
    #define DO_PRAGMA(x) PRAGMA_SUB(x)
#endif

//------------------------------------------------------------------------------
void lenet5(       DTYPE classes[10]
           , const DTYPE image  [32][32]
           #if !defined(__SYNTHESIS__)
           , const int rigor
           , const int verbose
           #endif
           )
{
    #if defined(__SYNTHESIS__)
        #pragma HLS INTERFACE m_axi depth=10 \
                              port = classes \
                              offset = slave \
                              bundle = data register
        #pragma HLS INTERFACE m_axi depth=1024 \
                              port = image \
                              offset = slave \
                              bundle = data register
        #pragma HLS INTERFACE s_axilite port = return \
                              bundle = ctl register
        #pragma HLS inline off
    #endif
          DTYPE     c1_out_data[6][28][28];
    const DTYPE   (*c1_in_data)[32][32]=(DTYPE (*)[32][32])image; // [1][32][32]
    const DTYPE   (*c1_kernel)[1][5][5]=(DTYPE (*)[1][5][5])conv1_weight; // [6][1][5][5]
    const DTYPE   (*c1_bias)=conv1_bias; // [6]
    const uint16_t  c1_out_size=28; //(((in_size-kernel_size+2*padding)/stride)+1));
    const uint16_t  c1_in_size=32;
    const uint8_t   c1_kernel_size=5;
    const uint16_t  c1_bias_size=6; //c1_out_channel
    const uint16_t  c1_in_channel=1;
    const uint16_t  c1_out_channel=6; //c1_bias_size
    const uint8_t   c1_stride=1;
    const uint8_t   c1_padding=0;

    Convolution2d<DTYPE> (
            (DTYPE *)c1_out_data
          , (DTYPE *)c1_in_data
          , (DTYPE *)c1_kernel
          , (DTYPE *)c1_bias
          ,          c1_out_size
          ,          c1_in_size
          ,          c1_kernel_size
          ,          c1_bias_size
          ,          c1_in_channel
          ,          c1_out_channel
          ,          c1_stride
          ,          c1_padding
          #if !defined(__SYNTHESIS__)
          ,          rigor
          ,          verbose
          #endif
    );

    #if EMBED_ReLU==0
          DTYPE     a1_out_data[6][28][28];
    const DTYPE   (*a1_in_data)[28][28]=c1_out_data; // [6][28][28]
    const uint32_t  a1_size=784; //c1_out_size*c1_out_size
    const uint32_t  a1_channel=6; //c1_out_channel

    ActivationReLu<DTYPE> (
            (DTYPE *)a1_out_data // contiguous: channel x size x size
          , (DTYPE *)a1_in_data  // contiguous: channel x size x size
          ,          a1_size     // number of elements per channel
          ,          a1_channel
          #if !defined(__SYNTHESIS__)
          ,          rigor
          ,          verbose
          #endif
    );
    #endif

          DTYPE     p1_out_data[6][14][14];
    #if EMBED_ReLU==1
    const DTYPE   (*p1_in_data)[28][28]=c1_out_data; // [6][28][28]
    #else
    const DTYPE   (*p1_in_data)[28][28]=a1_out_data; // [6][28][28]
    #endif
    const uint16_t  p1_out_size=14;
    const uint16_t  p1_in_size=28; //c1_out_size
    const uint8_t   p1_kernel_size=2;
    const uint8_t   p1_channel=6; //c1_out_channel
    const uint8_t   p1_stride=2;
    const uint8_t   p1_padding=0;
    const int       p1_ceil_mode=0;

    Pooling2dMax<DTYPE, EMBED_ReLU> ( // ReLU embedded
            (DTYPE *)p1_out_data    // out_channel x out_size x out_size
          , (DTYPE *)p1_in_data     // in_channel x in_size x in_size
          ,          p1_out_size    // only for square matrix
          ,          p1_in_size     // only for square matrix
          ,          p1_kernel_size // only for square matrix
          ,          p1_channel
          ,          p1_stride
          ,          p1_padding
          ,          p1_ceil_mode
          #if !defined(__SYNTHESIS__)
          ,          rigor
          ,          verbose
          #endif
    );

#if defined(DEBUG)
    printf("Pooling2dMax:\n");
    for (int oc=0; oc<p1_channel; oc++) {
        printf("[");
        for (int h=0; h<p1_out_size; h++) { // rows
            printf("[");
            for (int w=0; w<p1_out_size; w++) { // columns
                printf("%.4f", p1_out_data[oc][h][w]);
                if (w<p1_out_size) printf(",");
            }
            if (h<p1_out_size) printf("],\n");
            else               printf("]\n");
        }
        if (oc<p1_channel) printf("],\n");
        else               printf("]\n");
    }
#endif

          DTYPE     c2_out_data[16][10][10];
    const DTYPE   (*c2_in_data)[14][14]=p1_out_data; // [6][14][14]
    const DTYPE   (*c2_kernel)[6][5][5]=(DTYPE (*)[6][5][5])conv2_weight; // [16][6][5][5]
    const DTYPE   (*c2_bias)=conv2_bias; // [16]
    const uint16_t  c2_out_size=10;
    const uint16_t  c2_in_size=14; //p1_out_size
    const uint8_t   c2_kernel_size=5;
    const uint16_t  c2_bias_size=16; //c2_out_channel
    const uint16_t  c2_in_channel=6; //p1_channel
    const uint16_t  c2_out_channel=16; //c2_bias_size
    const uint8_t   c2_stride=1;
    const uint8_t   c2_padding=0;

    Convolution2d<DTYPE> (
            (DTYPE *)c2_out_data
          , (DTYPE *)c2_in_data
          , (DTYPE *)c2_kernel
          , (DTYPE *)c2_bias
          ,          c2_out_size
          ,          c2_in_size
          ,          c2_kernel_size
          ,          c2_bias_size
          ,          c2_in_channel
          ,          c2_out_channel
          ,          c2_stride
          ,          c2_padding
          #if !defined(__SYNTHESIS__)
          ,          rigor
          ,          verbose
          #endif
    );

    #if EMBED_ReLU==0
          DTYPE     a2_out_data[16][10][10];
    const DTYPE   (*a2_in_data)[10][10]=c2_out_data; // [6][14][14]
    const uint32_t  a2_size=100; //c2_out_size*c2_out_size
    const uint32_t  a2_channel=16; //c2_out_channel

    ActivationReLu<DTYPE> (
            (DTYPE *)a2_out_data // contiguous: channel x size x size
          , (DTYPE *)a2_in_data  // contiguous: channel x size x size
          ,          a2_size     // number of elements per channel
          ,          a2_channel
          #if !defined(__SYNTHESIS__)
          ,          rigor
          ,          verbose
          #endif
    );
    #endif

          DTYPE     p2_out_data[16][5][5];
    #if EMBED_ReLU==1
    const DTYPE   (*p2_in_data)[10][10]=c2_out_data; // [16][10][10]
    #else
    const DTYPE   (*p2_in_data)[10][10]=a2_out_data; // [16][10][10]
    #endif
    const uint16_t  p2_out_size=5;
    const uint16_t  p2_in_size=10; //c2_out_size
    const uint8_t   p2_kernel_size=2;
    const uint8_t   p2_channel=16; //c2_out_channel
    const uint8_t   p2_stride=2;
    const uint8_t   p2_padding=0;
    const int       p2_ceil_mode=0;

    Pooling2dMax<DTYPE, EMBED_ReLU> ( // ReLU embedded
            (DTYPE *)p2_out_data
          , (DTYPE *)p2_in_data
          ,          p2_out_size
          ,          p2_in_size
          ,          p2_kernel_size
          ,          p2_channel
          ,          p2_stride
          ,          p2_padding
          ,          p2_ceil_mode
          #if !defined(__SYNTHESIS__)
          ,          rigor
          ,          verbose
          #endif
    );

          DTYPE     f1_out_data[120];
    const DTYPE   (*f1_in_data)[5][5]=p2_out_data; // [16][5][5]
    const DTYPE   (*f1_weight)[400]=(DTYPE (*)[400])fc1_weight; // [120][400]
    const DTYPE   (*f1_bias)=fc1_bias; // [120]
    const uint16_t  f1_out_size=120;
    const uint16_t  f1_in_size=16*5*5; //400: p2_channel*p2_out_size*p2_out_size
    const uint16_t  f1_bias_size=120;

    Linear1d<DTYPE, EMBED_ReLU> ( // ReLU embedded
            (DTYPE *)f1_out_data
          , (DTYPE *)f1_in_data
          , (DTYPE *)f1_weight
          , (DTYPE *)f1_bias
          ,          f1_out_size
          ,          f1_in_size 
          ,          f1_bias_size
          #if !defined(__SYNTHESIS__)
          ,          rigor
          ,          verbose
          #endif
    );

    #if EMBED_ReLU==0
          DTYPE     a3_out_data[120];
    const DTYPE    *a3_in_data=f1_out_data; // [120]
    const uint32_t  a3_size=120; //f1_out_size
    const uint32_t  a3_channel=1; //1

    ActivationReLu<DTYPE> (
            (DTYPE *)a3_out_data // contiguous: channel x size x size
          , (DTYPE *)a3_in_data  // contiguous: channel x size x size
          ,          a3_size     // number of elements per channel
          ,          a3_channel
          #if !defined(__SYNTHESIS__)
          ,          rigor
          ,          verbose
          #endif
    );
    #endif

          DTYPE     f2_out_data[84];
    #if EMBED_ReLU==1
    const DTYPE   (*f2_in_data)=f1_out_data; // [120]
    #else
    const DTYPE   (*f2_in_data)=a3_out_data; // [120]
    #endif
    const DTYPE   (*f2_weight)[120]=(DTYPE (*)[120])fc2_weight; // [84][120]
    const DTYPE   (*f2_bias)=fc2_bias; // [84]
    const uint16_t  f2_out_size=84;
    const uint16_t  f2_in_size=120; //f1_out_size
    const uint16_t  f2_bias_size=84; //f2_out_size

    Linear1d<DTYPE, EMBED_ReLU> ( // ReLU embedded
            (DTYPE *)f2_out_data
          , (DTYPE *)f2_in_data
          , (DTYPE *)f2_weight
          , (DTYPE *)f2_bias
          ,          f2_out_size
          ,          f2_in_size 
          ,          f2_bias_size
          #if !defined(__SYNTHESIS__)
          ,          rigor
          ,          verbose
          #endif
    );

    #if EMBED_ReLU==0
          DTYPE     a4_out_data[84];
    const DTYPE    *a4_in_data=f2_out_data; // [84]
    const uint32_t  a4_size=84; //f2_out_size
    const uint32_t  a4_channel=1; //1

    ActivationReLu<DTYPE> (
            (DTYPE *)a4_out_data // contiguous: channel x size x size
          , (DTYPE *)a4_in_data  // contiguous: channel x size x size
          ,          a4_size     // number of elements per channel
          ,          a4_channel
          #if !defined(__SYNTHESIS__)
          ,          rigor
          ,          verbose
          #endif
    );
    #endif

          DTYPE   (*f3_out_data)=classes; // [10]
    #if EMBED_ReLU==1
    const DTYPE   (*f3_in_data)=f2_out_data; // [84]
    #else
    const DTYPE   (*f3_in_data)=a4_out_data; // [84]
    #endif
    const DTYPE   (*f3_weight)[84]=(DTYPE (*)[84])fc3_weight; // [10][84]
    const DTYPE   (*f3_bias)=fc3_bias; // [10]
    const uint16_t  f3_out_size=10;
    const uint16_t  f3_in_size=84;
    const uint16_t  f3_bias_size=10;

    Linear1d<DTYPE, 0> ( // ReLU not embedded
            (DTYPE *)f3_out_data
          , (DTYPE *)f3_in_data
          , (DTYPE *)f3_weight
          , (DTYPE *)f3_bias
          ,          f3_out_size
          ,          f3_in_size 
          ,          f3_bias_size
          #if !defined(__SYNTHESIS__)
          ,          rigor
          ,          verbose
          #endif
    );

}
//------------------------------------------------------------------------------
// Revision History
//
// 2020.10.05: Start by Ando Ki (adki@future-ds.com)
//------------------------------------------------------------------------------
