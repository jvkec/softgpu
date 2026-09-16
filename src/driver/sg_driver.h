#pragma once
// Kernel-mode-driver stand-in, exposed through a syscall-shaped interface.
//
// The runtime only ever talks to the driver through these three calls. In
// stage 8 they become thin wrappers around open()/ioctl()/close() on
// /dev/softgpu and the runtime does not change.

#ifdef __cplusplus
extern "C" {
#endif

// Returns a descriptor >= 0, or -errno.
int sg_drv_open(void);
// Returns 0 or -errno. `arg` points to the struct matching `req` (sg_ioctl.h).
int sg_drv_ioctl(int fd, unsigned int req, void* arg);
// Returns 0 or -errno.
int sg_drv_close(int fd);

#ifdef __cplusplus
}
#endif
