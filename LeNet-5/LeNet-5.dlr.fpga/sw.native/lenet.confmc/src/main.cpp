#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdint.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"
#if defined(HALF)
    #include "hls_half.h"
    #if !defined(DTYPE)
        #define DTYPE            half
        #define NBYTES_OF_DTYPE  2
    #endif
#elif defined(FIXED_POINT)
    #include "ap_fixed.h"
    #if !defined(DTYPE)
        #define DTYPE            ap_fixed<32,8>
        #define NBYTES_OF_DTYPE  4
    #endif
#else
    #if !defined(NBYTES_OF_DTYPE)
        #define NBYTES_OF_DTYPE  4
    #endif
#endif

#include "conapi.h"
#include "trx_axi_api.h"

unsigned int card_id=0;
con_Handle_t handle=NULL;

#      define MEM_WRITE(A, B)        BfmWrite(handle, (unsigned int)(A), (unsigned int*)&(B), 4, 1)
#      define MEM_READ(A, B)         BfmRead (handle, (unsigned int)(A), (unsigned int*)&(B), 4, 1)
#      define MEM_WRITE_G(A,D,S,L)   BfmWrite(handle, (A), (D), (S), (L))
#      define MEM_READ_G(A,D,S,L)    BfmRead (handle, (A), (D), (S), (L))

#define NUM_ROWS     32 // IMG_DMNIN
#define NUM_COLS     32 // IMG_DMNIN
#define SIZE_IMG    (NUM_ROWS*NUM_COLS) // num of pixels
#define NUM_CLASSES  10 // SFMX_SIZE
#define ADDR_CSR     0xC0000000
#define ADDR_IMG     0x00000000
#define ADDR_RESULT (ADDR_IMG+SIZE_IMG*NBYTES_OF_DTYPE)

// hw/hls/tcl.float/proj_lenet/solution1/impl/misc/drivers/lenet5_v1_0/src/xlenet5_hw.h
#define ADDR_CSR_AP_CTRL      (ADDR_CSR+0x00)
#define ADDR_CSR_GIE          (ADDR_CSR+0x04)
#define ADDR_CSR_IER          (ADDR_CSR+0x08)
#define ADDR_CSR_ISR          (ADDR_CSR+0x0c)
#define ADDR_CSR_AP_RETURN    (ADDR_CSR+0x10)
#define ADDR_CSR_CLASSES_DATA (ADDR_CSR+0x10)
#define ADDR_CSR_IMAGE_R_DATA (ADDR_CSR+0x18)


int lenet(char inputFileName[]);
int get_image_data(char inputFileName[], float greyData[SIZE_IMG]
                  ,int normalize=1, int revert=0);
void softmax(float p[NUM_CLASSES], DTYPE q[NUM_CLASSES]);

int main(int argc, char *argv[])
{
    if (argc!=2) {
        printf("Usage %s input_imapge\n", argv[0]);
        return 0;
    }

    if ((handle=conInit(card_id, CON_MODE_CMD, CONAPI_LOG_LEVEL_INFO))==NULL) {
         printf("cannot initialize CON-FMC\n");
         return 0;
    }
    struct _usb usb;
    conGetUsbInfo( handle, &usb); // Get USB related information
    con_MasterInfo_t gpif2mst_info; // Get GPIF2MST information
    if (conGetMasterInfo(handle, &gpif2mst_info)) {
        printf("cannot get gpif2mst info\n");
        return -1;
    }
    #ifndef SILENCE
    extern void confmc_info(struct _usb usb, con_MasterInfo_t gpif2mst_info);
    confmc_info(usb, gpif2mst_info);
    #endif //SILENCE

    unsigned int dataW, dataR;
    dataW = ADDR_IMG;
    MEM_WRITE(ADDR_CSR_IMAGE_R_DATA   , dataW);
    dataW = ADDR_RESULT;
    MEM_WRITE(ADDR_CSR_CLASSES_DATA, dataW);

    #ifdef DEBUG
    MEM_READ(ADDR_CSR_CLASSES_DATA, dataR);
    printf("%s() classes : 0x%08X\n", __func__, dataR);
    MEM_READ(ADDR_CSR_IMAGE_R_DATA, dataR);
    printf("%s() image : 0x%08X\n", __func__, dataR);
    #endif

    (void)lenet(argv[1]);

    return 0;
}

#define WAIT_FOR_READY {\
    int ap_idle, ap_idle_r;\
    unsigned int ap_addr;\
    ap_addr = ADDR_CSR_AP_CTRL;\
    while (1) {\
        MEM_READ(ap_addr, ap_idle_r);\
        ap_idle = (ap_idle_r >> 2) && 0x1;\
        if (ap_idle) break;\
    }}
#define GO_AND_WAIT_COMPLETE {\
    int ap_done, ap_done_r;\
    int ap_start, ap_data;\
    unsigned int ap_addr;\
    ap_addr = ADDR_CSR_AP_CTRL;\
    ap_data = 0x1;\
    MEM_WRITE(ap_addr, ap_data);\
    while (1) {\
        MEM_READ(ap_addr, ap_done_r);\
        ap_done = (ap_done_r >> 1) && 0x1;\
        if (ap_done) break;\
    }}

// 1. get image data (gray scale [0-1]
// 2. check IP is ready
// 3. write image data to the IP 
// 4. let IP go and wait for completion
// 5. read results from the IP
// 6. print results
#if defined(HALF)||defined(FIXED_POINT)
int lenet(char inputFileName[])
{
    float greyDataFloat[SIZE_IMG]; // note that it carries float [0~1]
    (void)get_image_data(inputFileName, greyDataFloat);
    DTYPE greyData[SIZE_IMG];
    for (int i=0; i<SIZE_IMG; i++) greyData[i] = static_cast<DTYPE>(greyDataFloat[i]);

    WAIT_FOR_READY
    
    MEM_WRITE_G(ADDR_IMG, (unsigned int*)&greyData[0], 4, SIZE_IMG/(4/NBYTES_OF_DTYPE));

    GO_AND_WAIT_COMPLETE

    DTYPE result[NUM_CLASSES];
    MEM_READ_G(ADDR_RESULT, (unsigned int*)&result[0], 4, NUM_CLASSES/(4/NBYTES_OF_DTYPE));

    float resultClasses[NUM_CLASSES];
    softmax(resultClasses, result);

    printf("    The probabilities of the digit being 0~9 are:\n");
    float maxVal=0.0;
    int   maxId=0;
    for (int i = 0; i < NUM_CLASSES; i++) {
        float val = *(float*)&resultClasses[i];
        if (maxVal<val) {
            maxVal = val;
            maxId  = i;
        }
    }
    for (int i = 0; i < NUM_CLASSES; i++) {
        float result = *(float*)&resultClasses[i];
        printf("        %d:  %f %s\n", i, result, (i==maxId) ? "*" : "");
    }

    return 0;
}
#elif defined(FIXED_POINT)
int lenet(char inputFileName[])
{
    float greyDataFloat[SIZE_IMG]; // note that it carries float [0~1]
    (void)get_image_data(inputFileName, greyDataFloat);
    DTYPE greyData[SIZE_IMG];
    for (int i=0; i<SIZE_IMG; i++) greyData[i] = static_cast<DTYPE>(greyDataFloat[i]);

    WAIT_FOR_READY
    
    MEM_WRITE_G(ADDR_IMG, (unsigned int*)&greyData[0], NBYTES_OF_DTYPE, SIZE_IMG);

    GO_AND_WAIT_COMPLETE

    DTYPE result[NUM_CLASSES];
    MEM_READ_G(ADDR_RESULT, (unsigned int*)&result[0], NBYTES_OF_DTYPE, NUM_CLASSES);

    float resultClasses[NUM_CLASSES];
    softmax(resultClasses, result);

    printf("    The probabilities of the digit being 0~9 are:\n");
    float maxVal=0.0;
    int   maxId=0;
    for (int i = 0; i < NUM_CLASSES; i++) {
        float val = *(float*)&resultClasses[i];
        if (maxVal<val) {
            maxVal = val;
            maxId  = i;
        }
    }
    for (int i = 0; i < NUM_CLASSES; i++) {
        float result = *(float*)&resultClasses[i];
        printf("        %d:  %f %s\n", i, result, (i==maxId) ? "*" : "");
    }

    return 0;
}
#else
int lenet(char inputFileName[])
{
    float greyDataFloat[SIZE_IMG]; // note that it carries float [0~1]
    (void)get_image_data(inputFileName, greyDataFloat);

    WAIT_FOR_READY
    
    MEM_WRITE_G(ADDR_IMG, (unsigned int*)&greyDataFloat[0], NBYTES_OF_DTYPE, SIZE_IMG);

    GO_AND_WAIT_COMPLETE

    float resultFloat[NUM_CLASSES];
    MEM_READ_G(ADDR_RESULT, (unsigned int*)&resultFloat[0], NBYTES_OF_DTYPE, NUM_CLASSES);

    float resultClasses[NUM_CLASSES];
    softmax(resultClasses, resultFloat);

    printf("    The probabilities of the digit being 0~9 are:\n");
    float maxVal=0.0;
    int   maxId=0;
    for (int i = 0; i < NUM_CLASSES; i++) {
        float val = *(float*)&resultClasses[i];
        if (maxVal<val) {
            maxVal = val;
            maxId  = i;
        }
    }
    for (int i = 0; i < NUM_CLASSES; i++) {
        float result = *(float*)&resultClasses[i];
        printf("        %d:  %f %s\n", i, result, (i==maxId) ? "*" : "");
    }

    return 0;
}
#endif

// normallized exponential function.
// how to deal with overflow of exponet.
// how to deal with 0 of denom.
//
// Testing vector
// Input Array  : 1.000000 4.200000 0.600000 1.230000 4.300000 1.200000 2.500000
// Softmax Array: 0.016590 0.406995 0.011121 0.020880 0.449799 0.020263 0.07435
void softmax(float p[NUM_CLASSES], DTYPE q[NUM_CLASSES])
{
#if 1
     float maxVal=-INFINITY;
     for (int i=0; i<NUM_CLASSES; i++) {
        float fval = (float)q[i];
        if (maxVal<fval) maxVal = fval;
     }
     float denom=0.0f;
     for (int i=0; i<NUM_CLASSES; i++) {
        float fval = (float)q[i] - maxVal;
        denom += expf(fval);
     }
     for (int i=0; i<NUM_CLASSES; i++) {
        float fval = (float)q[i] - maxVal;
        p[i] = expf(fval)/denom;
     }
#else
    int i;
    double m, sum, constant;

    m = -INFINITY;
    for (i = 0; i < size; ++i) {
        if (m < p[i]) {
            m = p[i];
        }
    }

    sum = 0.0;
    for (i = 0; i < size; ++i) {
        sum += expf(p[i] - m);
    }

    constant = m + logf(sum);
    for (i = 0; i < size; ++i) {
        q[i] = (DTYPE)(expf(p[i] - constant));
    }
#endif
}

// 1. Reads image data and converts gray color
// 2. Carries out normalization
int get_image_data( char inputFileName[], float greyData[SIZE_IMG]
                  , __attribute__((unused))int normalize, __attribute__((unused))int revert)
{
    int width, height, channels;
    unsigned char *img = stbi_load(inputFileName, &width, &height, &channels, 0);
    if (img==NULL) return -1;
printf("%s() %s w=%d h=%d c=%d\n", __func__, inputFileName, width, height, channels);
    if (channels==3) {
        for (int h=0; h<height; h++)
        for (int w=0; w<width ; w++) {
             unsigned int red   = img[h*width*3+w*3];
             unsigned int green = img[h*width*3+w*3+1];
             unsigned int blue  = img[h*width*3+w*3+2];
             float gray = 0.2126f * (float)red + 0.7152f * (float)green + 0.0722f * (float)blue;
             img[h*width+w] = (gray<0.0f) ? 0 : (gray>255.0f) ? 255 : (unsigned char)gray;
        }
    }
    if ((width!=NUM_COLS)||(height!=NUM_ROWS)) {
printf("%s() %s resized\n", __func__, inputFileName);
        unsigned char *imgRESIZED = (unsigned char*)malloc(NUM_COLS*NUM_ROWS*sizeof(unsigned char));
        stbir_resize_uint8(img, width, height, 0, imgRESIZED, NUM_COLS, NUM_ROWS, 0, 1);
        free(img);
        img = imgRESIZED;
        stbi_write_png("resized.png", NUM_COLS, NUM_ROWS, 1, img, NUM_COLS);
    }
    int num_blacks=0;
    float scale_min = -0.3635f;
    float scale_max = 3.2558f;
    for (int h=0; h<NUM_ROWS; h++)
    for (int w=0; w<NUM_COLS; w++) {
         greyData[(h*NUM_COLS)+w] = ((float)img[h*NUM_COLS+w]/255.0)*(scale_max-scale_min)+scale_min;
                                  // normaization: make [0-1]
         if (greyData[(h*NUM_COLS)+w]<0.5) num_blacks++;
    }
    if (num_blacks<(NUM_ROWS*NUM_COLS)/2) { // need revert
printf("%s() %s reverted\n", __func__, inputFileName);
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

void confmc_info(struct _usb usb, con_MasterInfo_t gpif2mst_info)
{
    printf("USB information\n");
    printf("    DevSpeed         =%d%cbps\n", (usb.speed>10000) ? usb.speed/10000
                                                                : usb.speed/10
                                            , (usb.speed>10000) ? 'G' : 'M');
    printf("    BulkMaxPktSizeOut=%d\n", usb.bulk_max_pkt_size_out);
    printf("    BulkMaxPktSizeIn =%d\n", usb.bulk_max_pkt_size_in );
    printf("    IsoMaxPktSizeOut =%d\n", usb.iso_max_pkt_size_out );
    printf("    IsoMaxPktSizeIn  =%d\n", usb.iso_max_pkt_size_in  );
    fflush(stdout);

    printf("gpif2mst information\n");
    printf("         version 0x%08X\n", gpif2mst_info.version);
    printf("         pclk_freq %d-Mhz (%s)\n", gpif2mst_info.clk_mhz
                                               , (gpif2mst_info.clk_inv)
                                               ? "inverted"
                                               : "not-inverted");
    printf("         DepthCu2f=%d, DepthDu2f=%d, DepthDf2u=%d\n"
                                 , gpif2mst_info.depth_cmd
                                 , gpif2mst_info.depth_u2f
                                 , gpif2mst_info.depth_f2u);
    fflush(stdout);
}
