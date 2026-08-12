/* hwjpeg — SG2002 硬件 JPEG 编码器封装（VENC）
 *
 * 供 Python ctypes 调用：
 *   hwjpeg_init(width, height, quality) → 0 成功
 *   hwjpeg_encode(bgr, w, h, &out_ptr, &out_len) → 0 成功（out 需 hwjpeg_free_out）
 *   hwjpeg_free_out(ptr)
 *   hwjpeg_close()
 *
 * 依赖板上 /usr/bin/dl_lib/libvenc.so + libsys.so（需 libatomic 预加载）。
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <dlfcn.h>

#include "cvi_venc.h"
#include "cvi_vb.h"
#include "cvi_sys.h"
#include "cvi_buffer.h"

/* ── 动态解析的函数指针 ── */
static CVI_S32 (*p_CVI_SYS_Init)(void);
static CVI_S32 (*p_CVI_SYS_Exit)(void);
static void *(*p_CVI_SYS_Mmap)(CVI_U64, CVI_U32);
static CVI_S32 (*p_CVI_SYS_Munmap)(void *, CVI_U32);
static CVI_S32 (*p_CVI_VB_SetConfig)(const VB_CONFIG_S *);
static CVI_S32 (*p_CVI_VB_Init)(void);
static CVI_S32 (*p_CVI_VB_Exit)(void);
static VB_BLK (*p_CVI_VB_GetBlock)(VB_POOL, CVI_U32);
static CVI_S32 (*p_CVI_VB_ReleaseBlock)(VB_BLK);
static CVI_U64 (*p_CVI_VB_Handle2PhysAddr)(VB_BLK);
static CVI_U32 (*p_CVI_VB_Handle2PoolId)(VB_BLK);
static CVI_S32 (*p_CVI_VENC_CreateChn)(VENC_CHN, const VENC_CHN_ATTR_S *);
static CVI_S32 (*p_CVI_VENC_DestroyChn)(VENC_CHN);
static CVI_S32 (*p_CVI_VENC_StartRecvFrame)(VENC_CHN, const VENC_RECV_PIC_PARAM_S *);
static CVI_S32 (*p_CVI_VENC_StopRecvFrame)(VENC_CHN);
static CVI_S32 (*p_CVI_VENC_SendFrame)(VENC_CHN, const VIDEO_FRAME_INFO_S *, CVI_S32);
static CVI_S32 (*p_CVI_VENC_GetStream)(VENC_CHN, VENC_STREAM_S *, CVI_S32);
static CVI_S32 (*p_CVI_VENC_ReleaseStream)(VENC_CHN, VENC_STREAM_S *);
static CVI_S32 (*p_CVI_VENC_SetJpegParam)(VENC_CHN, const VENC_JPEG_PARAM_S *);

static int g_initialized = 0;
static int g_width = 0, g_height = 0;

/* 中间件库搜索路径：优先仓库自带 cvi-libs/，回退系统 dl_lib */
static const char *LIB_DIRS[] = {
	"cvi-libs/",
	"/root/tennis-vnav/cvi-libs/",
	"/usr/bin/dl_lib/",
	NULL,
};

static void *dlopen_try(const char *name)
{
	for (int i = 0; LIB_DIRS[i]; i++) {
		char path[256];
		snprintf(path, sizeof(path), "%s%s", LIB_DIRS[i], name);
		void *h = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
		if (h)
			return h;
	}
	return NULL;
}

static int load_symbols(void)
{
	void *h_atomic = dlopen_try("libatomic.so.1");
	if (!h_atomic)
		h_atomic = dlopen("libatomic.so.1", RTLD_NOW | RTLD_GLOBAL);
	if (!h_atomic) {
		fprintf(stderr, "[hwjpeg] dlopen libatomic: %s\n", dlerror());
		/* 非致命：若调用进程已预加载则继续 */
	}

	void *h_sys = dlopen_try("libsys.so");
	if (!h_sys) {
		fprintf(stderr, "[hwjpeg] libsys.so 加载失败: %s\n", dlerror());
		return -1;
	}
	void *h_venc = dlopen_try("libvenc.so");
	if (!h_venc) {
		fprintf(stderr, "[hwjpeg] libvenc.so 加载失败: %s\n", dlerror());
		return -1;
	}

#define LOAD(h, sym) do { \
	*(void **)(&p_##sym) = dlsym(h, #sym); \
	if (!p_##sym) { fprintf(stderr, "[hwjpeg] dlsym %s: %s\n", #sym, dlerror()); return -1; } \
} while (0)

	LOAD(h_sys, CVI_SYS_Init);
	LOAD(h_sys, CVI_SYS_Exit);
	LOAD(h_sys, CVI_SYS_Mmap);
	LOAD(h_sys, CVI_SYS_Munmap);
	LOAD(h_sys, CVI_VB_SetConfig);
	LOAD(h_sys, CVI_VB_Init);
	LOAD(h_sys, CVI_VB_Exit);
	LOAD(h_sys, CVI_VB_GetBlock);
	LOAD(h_sys, CVI_VB_ReleaseBlock);
	LOAD(h_sys, CVI_VB_Handle2PhysAddr);
	LOAD(h_sys, CVI_VB_Handle2PoolId);
	LOAD(h_venc, CVI_VENC_CreateChn);
	LOAD(h_venc, CVI_VENC_DestroyChn);
	LOAD(h_venc, CVI_VENC_StartRecvFrame);
	LOAD(h_venc, CVI_VENC_StopRecvFrame);
	LOAD(h_venc, CVI_VENC_SendFrame);
	LOAD(h_venc, CVI_VENC_GetStream);
	LOAD(h_venc, CVI_VENC_ReleaseStream);
	LOAD(h_venc, CVI_VENC_SetJpegParam);
	return 0;
}

int hwjpeg_init(int width, int height, int quality)
{
	CVI_S32 s32Ret;

	if (g_initialized)
		return 0;

	if (load_symbols() != 0)
		return -1;

	/* 官方顺序（SAMPLE_COMM_SYS_Init）：
	 *   先清理 → VB_SetConfig → VB_Init → 最后 SYS_Init
	 * 注意：此板 SDK 要求 u32MaxPoolCnt ≥ 1（0 池返回 VB_ILLEGAL_PARAM） */
	p_CVI_SYS_Exit();
	p_CVI_VB_Exit();

	VB_CONFIG_S stVbConf;
	memset(&stVbConf, 0, sizeof(stVbConf));
	stVbConf.u32MaxPoolCnt = 1;
	stVbConf.astCommPool[0].u32BlkSize = (CVI_U32)width * height * 3;
	stVbConf.astCommPool[0].u32BlkCnt = 4;
	stVbConf.astCommPool[0].enRemapMode = VB_REMAP_MODE_NONE;
	strcpy(stVbConf.astCommPool[0].acName, "common");

	s32Ret = p_CVI_VB_SetConfig(&stVbConf);
	if (s32Ret != CVI_SUCCESS) {
		fprintf(stderr, "[hwjpeg] CVI_VB_SetConfig: %d\n", s32Ret);
		return -1;
	}
	s32Ret = p_CVI_VB_Init();
	if (s32Ret != CVI_SUCCESS) {
		fprintf(stderr, "[hwjpeg] CVI_VB_Init: %d\n", s32Ret);
		return -1;
	}
	s32Ret = p_CVI_SYS_Init();
	if (s32Ret != CVI_SUCCESS) {
		fprintf(stderr, "[hwjpeg] CVI_SYS_Init: %d\n", s32Ret);
		return -1;
	}

	VENC_CHN_ATTR_S stAttr;
	memset(&stAttr, 0, sizeof(stAttr));
	stAttr.stVencAttr.enType = PT_JPEG;
	stAttr.stVencAttr.u32MaxPicWidth = width;
	stAttr.stVencAttr.u32MaxPicHeight = height;
	stAttr.stVencAttr.u32PicWidth = width;
	stAttr.stVencAttr.u32PicHeight = height;
	stAttr.stVencAttr.u32BufSize = (CVI_U32)width * height;
	stAttr.stVencAttr.u32Profile = 0;
	stAttr.stVencAttr.bByFrame = CVI_TRUE;
	stAttr.stVencAttr.stAttrJpege.bSupportDCF = CVI_FALSE;
	stAttr.stVencAttr.stAttrJpege.stMPFCfg.u8LargeThumbNailNum = 0;
	stAttr.stVencAttr.stAttrJpege.enReceiveMode = VENC_PIC_RECEIVE_SINGLE;
	stAttr.stGopAttr.enGopMode = VENC_GOPMODE_NORMALP;
	stAttr.stGopAttr.stNormalP.s32IPQpDelta = 0;

	s32Ret = p_CVI_VENC_CreateChn(0, &stAttr);
	if (s32Ret != CVI_SUCCESS) {
		fprintf(stderr, "[hwjpeg] CVI_VENC_CreateChn: %d\n", s32Ret);
		return -1;
	}

	VENC_JPEG_PARAM_S stJpegParam;
	memset(&stJpegParam, 0, sizeof(stJpegParam));
	if (quality < 1) quality = 1;
	if (quality > 99) quality = 99;
	stJpegParam.u32Qfactor = quality;
	s32Ret = p_CVI_VENC_SetJpegParam(0, &stJpegParam);
	if (s32Ret != CVI_SUCCESS)
		fprintf(stderr, "[hwjpeg] SetJpegParam: %d (继续)\n", s32Ret);

	VENC_RECV_PIC_PARAM_S stRecv;
	stRecv.s32RecvPicNum = -1;  /* 连续接收 */
	s32Ret = p_CVI_VENC_StartRecvFrame(0, &stRecv);
	if (s32Ret != CVI_SUCCESS) {
		fprintf(stderr, "[hwjpeg] CVI_VENC_StartRecvFrame: %d\n", s32Ret);
		return -1;
	}

	g_initialized = 1;
	g_width = width;
	g_height = height;
	return 0;
}

int hwjpeg_encode(const unsigned char *bgr, int width, int height,
		  unsigned char **out_ptr, int *out_len)
{
	CVI_S32 s32Ret;
	VIDEO_FRAME_INFO_S stFrameInfo;
	VIDEO_FRAME_S *pstVFrame;
	VB_BLK blk;
	CVI_U64 u64Phys;
	CVI_U8 *pVir;
	VENC_STREAM_S stStream;
	VENC_PACK_S stPack;
	CVI_U32 stride, ysize, uvsize, blksize;
	CVI_U8 *pY, *pUV;

	if (!g_initialized || !bgr || !out_ptr || !out_len)
		return -1;

	/* VENC 不接受 RGB 输入，转 NV12 + 显式双平面地址。
	 * 只填 plane 0 时硬件会从垃圾地址读色度 → 整帧花屏。 */
	stride = width;
	ysize = stride * height;
	uvsize = stride * height / 2;
	blksize = ysize + uvsize;

	blk = p_CVI_VB_GetBlock(VB_INVALID_POOLID, blksize);
	if (blk == VB_INVALID_HANDLE) {
		fprintf(stderr, "[hwjpeg] GetBlock failed\n");
		return -1;
	}

	u64Phys = p_CVI_VB_Handle2PhysAddr(blk);
	pVir = (CVI_U8 *)p_CVI_SYS_Mmap(u64Phys, blksize);
	if (!pVir || pVir == (void *)-1) {
		fprintf(stderr, "[hwjpeg] SYS_Mmap failed\n");
		p_CVI_VB_ReleaseBlock(blk);
		return -1;
	}

	/* YUYV → NV12（纯字节重排，无色彩运算）
	 * YUYV 每 4 字节 = [Y0 U Y1 V]。
	 * VB 缓冲是 uncached 映射，先在缓存内存转换，再 memcpy 进 VB。 */
	pY = pVir;
	pUV = pVir + ysize;
	{
		CVI_U8 *tmp = malloc(blksize);
		CVI_U8 *tY = tmp;
		CVI_U8 *tUV = tmp + ysize;
		if (!tmp) {
			p_CVI_SYS_Munmap(pVir, blksize);
			p_CVI_VB_ReleaseBlock(blk);
			return -1;
		}

		for (int y = 0; y < height; y += 2) {
			const CVI_U8 *row0 = bgr + y * width * 2;       /* YUYV 行 */
			const CVI_U8 *row1 = row0 + width * 2;
			CVI_U8 *y0 = tY + y * stride;
			CVI_U8 *y1 = y0 + stride;
			CVI_U8 *uv = tUV + (y / 2) * stride;
			for (int x = 0; x < width; x += 2) {
				y0[x]     = row0[x * 2];
				y0[x + 1] = row0[x * 2 + 2];
				y1[x]     = row1[x * 2];
				y1[x + 1] = row1[x * 2 + 2];
				uv[x]     = row0[x * 2 + 1];   /* U（丢弃奇数行，垂直 1/2 抽样） */
				uv[x + 1] = row0[x * 2 + 3];   /* V */
			}
		}
		memcpy(pVir, tmp, blksize);
		free(tmp);
	}

	memset(&stFrameInfo, 0, sizeof(stFrameInfo));
	pstVFrame = &stFrameInfo.stVFrame;
	pstVFrame->u32Width = width;
	pstVFrame->u32Height = height;
	pstVFrame->enPixelFormat = PIXEL_FORMAT_NV12;
	pstVFrame->enVideoFormat = VIDEO_FORMAT_LINEAR;
	pstVFrame->enDynamicRange = DYNAMIC_RANGE_SDR8;
	pstVFrame->enCompressMode = COMPRESS_MODE_NONE;
	pstVFrame->u32Stride[0] = stride;
	pstVFrame->u32Stride[1] = stride;
	pstVFrame->u64PhyAddr[0] = u64Phys;
	pstVFrame->u64PhyAddr[1] = u64Phys + ysize;
	pstVFrame->pu8VirAddr[0] = pVir;
	pstVFrame->pu8VirAddr[1] = pVir + ysize;
	pstVFrame->u32Length[0] = ysize;
	pstVFrame->u32Length[1] = uvsize;
	stFrameInfo.u32PoolId = p_CVI_VB_Handle2PoolId(blk);

	s32Ret = p_CVI_VENC_SendFrame(0, &stFrameInfo, 100);
	if (s32Ret != CVI_SUCCESS) {
		fprintf(stderr, "[hwjpeg] SendFrame: %d\n", s32Ret);
		goto fail;
	}

	memset(&stStream, 0, sizeof(stStream));
	memset(&stPack, 0, sizeof(stPack));
	stStream.pstPack = &stPack;
	s32Ret = p_CVI_VENC_GetStream(0, &stStream, 200);
	if (s32Ret != CVI_SUCCESS) {
		fprintf(stderr, "[hwjpeg] GetStream: %d\n", s32Ret);
		goto fail;
	}

	if (stStream.u32PackCount < 1 || stPack.u32Len < 2) {
		fprintf(stderr, "[hwjpeg] empty stream (packs=%u len=%u)\n",
			stStream.u32PackCount, stPack.u32Len);
		p_CVI_VENC_ReleaseStream(0, &stStream);
		goto fail;
	}

	/* 有效 JPEG 数据: pu8Addr + u32Offset, 长度 u32Len - u32Offset */
	{
		CVI_U32 jpeg_len = stPack.u32Len - stPack.u32Offset;
		CVI_U8 *src = stPack.pu8Addr + stPack.u32Offset;
		unsigned char *out = malloc(jpeg_len);
		if (!out) {
			p_CVI_VENC_ReleaseStream(0, &stStream);
			goto fail;
		}
		memcpy(out, src, jpeg_len);
		*out_ptr = out;
		*out_len = (int)jpeg_len;
	}

	p_CVI_VENC_ReleaseStream(0, &stStream);
	p_CVI_SYS_Munmap(pVir, blksize);
	p_CVI_VB_ReleaseBlock(blk);
	return 0;

fail:
	p_CVI_SYS_Munmap(pVir, blksize);
	p_CVI_VB_ReleaseBlock(blk);
	return -1;
}

void hwjpeg_free_out(unsigned char *ptr)
{
	if (ptr)
		free(ptr);
}

void hwjpeg_close(void)
{
	if (!g_initialized)
		return;
	p_CVI_VENC_StopRecvFrame(0);
	p_CVI_VENC_DestroyChn(0);
	p_CVI_SYS_Exit();
	p_CVI_VB_Exit();
	g_initialized = 0;
}
