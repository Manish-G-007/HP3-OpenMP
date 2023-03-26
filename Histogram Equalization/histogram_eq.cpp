#include <bits/stdc++.h>
#include <omp.h>
#include <png.h>

using namespace std;

int width, height, n_threads;
png_byte color_type;
png_byte bit_depth;
png_bytep *row_pointers = NULL;


/*------------------------ Function used for reading png file---------------------------*/
void read_png_file(char *filename) 
{
  FILE *fp = fopen(filename, "rb");
  png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if(!png) abort();

  png_infop info = png_create_info_struct(png);
  if(!info) abort();

  if(setjmp(png_jmpbuf(png))) abort();

  png_init_io(png, fp);
  png_read_info(png, info);
  width      = png_get_image_width(png, info);
  height     = png_get_image_height(png, info);
  color_type = png_get_color_type(png, info);
  bit_depth  = png_get_bit_depth(png, info);


  if(bit_depth == 16)
    png_set_strip_16(png);

  if(color_type == PNG_COLOR_TYPE_PALETTE)
    png_set_palette_to_rgb(png);

  // PNG_COLOR_TYPE_GRAY_ALPHA is always 8 or 16bit depth.
  if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
    png_set_expand_gray_1_2_4_to_8(png);

  if(png_get_valid(png, info, PNG_INFO_tRNS))
    png_set_tRNS_to_alpha(png);

  // These color_type don't have an alpha channel then fill it with 0xff.
  if(color_type == PNG_COLOR_TYPE_RGB ||
     color_type == PNG_COLOR_TYPE_GRAY ||
     color_type == PNG_COLOR_TYPE_PALETTE)
    png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

  if(color_type == PNG_COLOR_TYPE_GRAY ||
     color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
    png_set_gray_to_rgb(png);

  png_read_update_info(png, info);
  if (row_pointers) abort();

  row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
  for(int y = 0; y < height; y++) {
    row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png,info));
  }
  png_read_image(png, row_pointers);
  fclose(fp);
  png_destroy_read_struct(&png, &info, NULL);
}


/*------------------------ Function used for writing png file---------------------------*/
void write_png_file(char *filename) 
{
  int y;

  FILE *fp = fopen(filename, "wb");
  if(!fp) abort();

  png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!png) abort();

  png_infop info = png_create_info_struct(png);
  if (!info) abort();

  if (setjmp(png_jmpbuf(png))) abort();

  png_init_io(png, fp);

  // Output is 8bit depth, RGBA format.
  png_set_IHDR(
    png,
    info,
    width, height,
    8,
    PNG_COLOR_TYPE_RGBA,
    PNG_INTERLACE_NONE,
    PNG_COMPRESSION_TYPE_DEFAULT,
    PNG_FILTER_TYPE_DEFAULT
  );
  png_write_info(png, info);

  if (!row_pointers) abort();

  png_write_image(png, row_pointers);
  png_write_end(png, NULL);

  for(int y = 0; y < height; y++) {
    free(row_pointers[y]);
  }
  free(row_pointers);

  fclose(fp);
  if (png && info)
          png_destroy_write_struct(&png, &info);
}

/*------------------------ Function used for Histogram equilization---------------------------*/
void hist_equilization() 
{

  cout << "No of threads: " << n_threads << endl;
  int ipix=pow(2,bit_depth);

  // Intialisation of histogram,transfer function
  int his_r[ipix],his_g[ipix],his_b[ipix],his_a[ipix];
  float t_r[ipix],t_g[ipix],t_b[ipix],t_a[ipix];


  for(int i=0; i<ipix; i++)
  {
      his_r[i] = 0; 
      his_g[i] = 0;
      his_b[i] = 0; 
      his_a[i] = 0;
      t_r[i] = 0.0; 
      t_g[i] = 0.0;
      t_b[i] = 0.0; 
      t_a[i] = 0.0; 
  }
    
  // omp_set_dynamic(0);
  omp_set_num_threads(n_threads);

  double st= omp_get_wtime();


  /* calculating histogram*/
  #pragma omp parallel for reduction(+:his_r, his_g, his_b, his_a)
    for(int y = 0; y < height; y++) {
      png_bytep row = row_pointers[y];
      for(int x = 0; x < width; x++) {
        png_bytep px = &(row[x * 4]);
        his_r[px[0]]++;
        his_g[px[1]]++;
        his_b[px[2]]++;
        his_a[px[3]]++;

      }
    }
  /* Histogram equalization*/
  #pragma omp parallel for
    for(int i = 0; i < ipix; i++){
        for(int j = 0; j < i+1; j++){
            t_r[i] += (ipix-1)*((float)his_r[j])/(height*width);
            t_g[i] += (ipix-1)*((float)his_g[j])/(height*width);
            t_b[i] += (ipix-1)*((float)his_b[j])/(height*width);
            t_a[i] += (ipix-1)*((float)his_a[j])/(height*width);
        }

    }

  
  #pragma omp parallel for collapse(2)
    for(int y = 0; y < height; y++) {
      for(int x = 0; x < width; x++) {
        png_bytep px = &(row_pointers[y][x * 4]);
        int of=x*4;
        row_pointers[y][of]=floor(t_r[px[0]]);
        row_pointers[y][of+1]=floor(t_g[px[1]]);
        row_pointers[y][of+2]=floor(t_b[px[2]]);
        row_pointers[y][of+3]=floor(t_a[px[3]]);
      }
    }

    double et = omp_get_wtime();
    cout << "Wall Time Elapsed : " << (et-st)  << endl;

}

int main(int argc, char *argv[]) 
{
  // if(argc != 3) abort();
  n_threads=atoi(argv[3]);
  read_png_file(argv[1]);  
  hist_equilization();
  write_png_file(argv[2]);

  return 0;
}

