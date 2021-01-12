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
#include <getopt.h>
#include <ctype.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

#if !defined(DTYPE)
#define DTYPE float
#endif

//------------------------------------------------------------------------------
static char version[]="2020.10.04";
static int rigor=0;
static int verbose=0;

#define MAX_IMAGES  10
#define MAX_LENG    128
char image_files[MAX_IMAGES][MAX_LENG];

#define NUM_CLASSES  10
#define IMAGE_WIDTH  32
#define IMAGE_HEIGHT 32
#define IMAGE_SIZE  (IMAGE_HEIGHT*IMAGE_WIDTH)

#define NUM_COLS     IMAGE_WIDTH
#define NUM_ROWS     IMAGE_HEIGHT
#define SIZE_IMG     IMAGE_SIZE

//------------------------------------------------------------------------------
static int  help( int argc, char **argv );
static int  arg_parser( int argc, char** argv );
static int  get_image_data( char *inputFileName
                          , float greyData[IMAGE_SIZE]
                          , int normalize
                          , int revert);

extern void lenet5(       DTYPE classes[10]
                  , const DTYPE image[32][32]
                  #if !defined(__SYNTHESIS__)
                  , const int rigor
                  , const int verbose
                  #endif
                  );

static void softmax( float p[NUM_CLASSES], DTYPE q[NUM_CLASSES]);

//------------------------------------------------------------------------------
int main( int argc, char* argv[])
{
    int num_images = arg_parser(argc, argv);
    
    int idx;
    for (idx=0; idx<num_images; idx++) {
        float classes[NUM_CLASSES];
        float image[IMAGE_HEIGHT][IMAGE_WIDTH];

        if (get_image_data( image_files[idx]
                          , (float *)image
                          , 0  // normalize
                          , 2))// revert if required
        {
            printf("%s() ERROR while reading image %s\n", __func__, image_files[idx]);
            continue;
        }
        
        lenet5( classes
              , image
              #if !defined(__SYNTHESIS__)
              , rigor
              , verbose
              #endif
              );

        float classes_softmax[NUM_CLASSES];
        softmax(classes_softmax, classes);
        printf("[%s]\n", image_files[idx]);
        int idy;
        for (idy=0; idy<NUM_CLASSES; idy++) {
             printf("\t[%d]: %f:%8.3f\n", idy, classes_softmax[idy], classes[idy]);
        }
    }
    
    return 0;
}

//------------------------------------------------------------------------------
void softmax(float p[NUM_CLASSES], DTYPE q[NUM_CLASSES])
{
     double denom=0.0f;
     for (int i=0; i<NUM_CLASSES; i++) {
        double fval = (double)q[i];
        denom += exp(fval);
     }
     for (int i=0; i<NUM_CLASSES; i++) {
        double fval = (double)q[i];
        p[i] = exp(fval)/denom;
     }
}

//------------------------------------------------------------------------------
// 1. Reads image data and converts gray color if required
// 2. Revert black and white if required
// 3. Carries out normalization if 'normalize' is 1
// 4. Return 0 on success
int get_image_data( char *inputFileName
                  , float greyData[SIZE_IMG]
                  , __attribute__((unused))int normalize
                  , int revert) // 0: do not apply
                                // 1: force to revert
                                // 2: do revert if required
{
    int width, height, channels;
    unsigned char *img = stbi_load(inputFileName, &width, &height, &channels, 0);
    if (img==NULL) return -1;
    if (channels==3) {
        for (int h=0; h<height; h++)
        for (int w=0; w<width ; w++) {
             unsigned int red   = img[h*width*3+w*3];
             unsigned int green = img[h*width*3+w*3+1];
             unsigned int blue  = img[h*width*3+w*3+2];
             float gray = 0.2126f * (float)red + 0.7152f * (float)green + 0.0722f * (float)blue;
             img[h*width+w] = (gray<0.0f) ? 0 : (gray>255.0f) ? 255 : (unsigned char)gray;
        }
    } else if (channels==4) { // including alpa
        for (int h=0; h<height; h++)
        for (int w=0; w<width ; w++) {
             unsigned int red   = img[h*width*4+w*4];
             unsigned int green = img[h*width*4+w*4+1];
             unsigned int blue  = img[h*width*4+w*4+2];
             float gray = 0.2126f * (float)red + 0.7152f * (float)green + 0.0722f * (float)blue;
             img[h*width+w] = (gray<0.0f) ? 0 : (gray>255.0f) ? 255 : (unsigned char)gray;
        }
    } else if (channels!=1) {
        printf("%s() ERROR cannot handle %d-channel\n", __func__, channels);
        return -1;
    }
    if ((width!=NUM_COLS)||(height!=NUM_ROWS)) {
        unsigned char *imgRESIZED = (unsigned char*)malloc(NUM_COLS*NUM_ROWS*sizeof(unsigned char));
        stbir_resize_uint8(img, width, height, 0, imgRESIZED, NUM_COLS, NUM_ROWS, 0, 1);
        free(img);
        img = imgRESIZED;
        stbi_write_png("resized.png", NUM_COLS, NUM_ROWS, 1, img, NUM_COLS);
    }
    int num_blacks=0;
#if 1
    for (int h=0; h<NUM_ROWS; h++)
    for (int w=0; w<NUM_COLS; w++) {
         float pvalue = (((float)img[h*NUM_COLS+w])/255.0);
         greyData[(h*NUM_COLS)+w] = pvalue;
         if (pvalue<0.5) num_blacks++;
    }
#else
    float scale_min = -0.3635f;
    float scale_max = 3.2558f;
    for (int h=0; h<NUM_ROWS; h++)
    for (int w=0; w<NUM_COLS; w++) {
         greyData[(h*NUM_COLS)+w] = ((float)img[h*NUM_COLS+w]/255.0)*(scale_max-scale_min)+scale_min;
                                  // normaization: make [0-1]
         if (greyData[(h*NUM_COLS)+w]<0.5) num_blacks++;
    }
#endif
    if ((revert==1)||((revert==2)&&(num_blacks<(NUM_ROWS*NUM_COLS)/2))) { // revert
        for (int h=0; h<NUM_ROWS; h++)
        for (int w=0; w<NUM_COLS; w++) {
             greyData[(h*NUM_COLS)+w] = 1.0 - greyData[(h*NUM_COLS)+w];
        }
        for (int h=0; h<NUM_ROWS; h++)
        for (int w=0; w<NUM_COLS; w++) {
             img[(h*NUM_COLS)+w] = (unsigned char)(greyData[(h*NUM_COLS)+w]*255.0);
        }
        stbi_write_png("reverted.png", NUM_COLS, NUM_ROWS, 1, img, NUM_COLS);
    }
    stbi_image_free(img);
    return 0;
}

//------------------------------------------------------------------------------
static int arg_parser( int argc, char** argv )
{
    int   cnt_images=0;
    
    static struct option long_options[] = {
           {"rigor"        , no_argument      , 0, 'r'},
           {"verbose"      , no_argument      , 0, 'b'},
    
           {"help"         , no_argument      , 0, 'h'},
           {"version"      , no_argument      , 0, 'v'},
           {0              , 0                , 0,  0 }
    };
    
    int longind=0;
    while (optind<argc) {
        int opt=getopt_long(argc,argv,"srbhv",long_options,&longind);
        if (opt==-1) break;
        switch (opt) {
        case 0:    printf("option %s", long_options[longind].name);
                  if (optarg) printf(" with arg %s", optarg);
                  printf("\n");
                  break;
        case 'r': rigor=1;
                  break;
        case 'b': verbose=1;
                  break;
        case 'h': help(argc, argv);
                  return 0;
        case 'v': printf("%s\n", version);
                  return 0;
        case '?': if (optopt=='c') {
                      printf("%s -%c requires an argument.\n", argv[0], optopt);
                  } else if (optopt=='d') {
                      printf("%s -%c requires an argument.\n", argv[0], optopt);
                      return -1;
                  } else if (isprint(optopt)) {
                      printf("%s -%c unknown\n", argv[0], optopt);
                  } else {
                      printf("%s \\x%x unknown character\n", argv[0], optopt);
                  }
                  return -1;
                  break;
        default:  printf("%s unknown option -%c\n", argv[0], opt);
                  return -1;
                  break;
        }
    }
    while (optind<argc) {
        if (cnt_images<MAX_IMAGES) {
            if (strlen(argv[optind])<MAX_LENG) {
                strcpy(image_files[cnt_images], argv[optind]);
                cnt_images++;
            }
        }
        optind++;
    }
    return cnt_images;
}

//------------------------------------------------------------------------------
static int help( int argc, char **argv )
{
       printf("[Usage] %s [-rb] [-hv] image [image ...]\n", argv[0]);
       printf("            -r|--rigor    check rigorously\n");
       printf("            -b|--verbose  set verbose mode\n");
       printf("            -h            help\n");
       printf("            -v            version\n");
       return 0;
}

//------------------------------------------------------------------------------
// Revision History
//
// 2020.10.05: Start by Ando Ki (adki@future-ds.com)
//------------------------------------------------------------------------------
