#include <assert.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "stable-diffusion-wrapper.h"
#include "stable-diffusion.cpp/stable-diffusion.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stable-diffusion.cpp/examples/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stable-diffusion.cpp/examples/stb_image_write.h"

extern "C" struct StableDiffusion;

extern "C" __declspec(dllexport) StableDiffusion* StableDiffusion_Create(int n_threads,
                                            bool vae_decode_only,
                                            bool free_params_immediately,
                                            RNGType rng_type)
                                            {
                                                return new StableDiffusion(n_threads, vae_decode_only, free_params_immediately, rng_type);
                                            }

extern "C" __declspec(dllexport) bool StableDiffusion_LoadFromFile(StableDiffusion* sd, const char* file_path)
{
    return sd->load_from_file(file_path);
}

extern "C" __declspec(dllexport) int StableDiffusion_Txt2Img_Path(StableDiffusion* sd,
                                           const char* prompt,
                                           const char* negative_prompt,
                                           float cfg_scale,
                                           int width,
                                           int height,
                                           SampleMethod sample_method,
                                           int sample_steps,
                                           int64_t seed,
                                           const char* path)
{
    std::vector<uint8_t> dataToCallback = sd->txt2img(prompt, negative_prompt, cfg_scale, width, height, sample_method, sample_steps, seed);
    stbi_write_png(path, width, height, 3, dataToCallback.data(), 0);
    return 0;
}

extern "C" __declspec(dllexport) int StableDiffusion_Img2Img_Path(StableDiffusion* sd,
                                           const char* init_img,
                                           const char* prompt,
                                           const char* negative_prompt,
                                           float cfg_scale,
                                           int width,
                                           int height,
                                           SampleMethod sample_method,
                                           int sample_steps,
                                           float strength,
                                           int64_t seed,
                                           const char* path)
{
    int c = 0;
    std::vector<uint8_t> initimg;
    unsigned char* img_data = stbi_load(init_img, &width, &height, &c, 3);
        if (img_data == NULL) {
            fprintf(stderr, "load image from '%s' failed\n", init_img);
            return 1;
        }
        if (c != 3) {
            fprintf(stderr, "input image must be a 3 channels RGB image, but got %d channels\n", c);
            free(img_data);
            return 1;
        }
        if (width <= 0 || width % 64 != 0) {
            fprintf(stderr, "error: the width of image must be a multiple of 64\n");
            free(img_data);
            return 1;
        }
        if (height <= 0 || height % 64 != 0) {
            fprintf(stderr, "error: the height of image must be a multiple of 64\n");
            free(img_data);
            return 1;
        }
    initimg.assign(img_data, img_data + (width * height * c));
    std::vector<uint8_t> img = sd->img2img(initimg,
                         prompt,
                         negative_prompt,
                         cfg_scale,
                         width,
                         height,
                         sample_method,
                         sample_steps,
                         strength,
                         seed);
    stbi_write_png(path, width, height, 3, img.data(), 0);
    return 0;
}
