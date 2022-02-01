#include <stdint.h>

void main(uint8_t* im, int32_t* h, int32_t* w, _Bool* mask, uint8_t* color) {
    
    int32_t x, y;
    
    #pragma omp parallel for
    for (y=0; y<h[0]; ++y) {
        for (x=0; x<w[0]; ++x) {
            if (mask[y*w[0] + x]) {
                int32_t pos = 3 * (y*w[0] + x);
                im[pos+0] = (im[pos+0]/2) + (color[0]/2);
                im[pos+1] = (im[pos+1]/2) + (color[1]/2);
                im[pos+2] = (im[pos+2]/2) + (color[2]/2);
            }
        }
    }
}