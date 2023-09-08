#ifndef __STABLE_DIFFUSION_WRAPPER_H__
#define __STABLE_DIFFUSION_WRAPPER_H__

#include <memory>
#include <vector>

extern "C"
{
    typedef void (*stable_diffusion_callback)(uint8_t* pdata, uint32_t length);
}

#endif  // __STABLE_DIFFUSION_WRAPPER_H__