import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TRITON_INTERPRET'] ='1'
import torch
import triton
import triton.language as tl

@triton.jit
def add_one(input, output,stride_bs,stride_h,stride_seq,stride_d):
    block_id = tl.program_id(0)
    bs = 12
    h = 3
    offset_bsh = bs*stride_bs+h*stride_h
    block_x = tl.make_block_ptr(
        base= input+offset_bsh,
        shape = (8,4),
        strides=(stride_seq,stride_d),
        offsets=(block_id*2,0),#block_id * block_size
        block_shape=(2,2),
        order=(1,0),
    )
    block_y = tl.make_block_ptr(
        base= output+offset_bsh,
        shape = (8,4),
        strides=(stride_seq,stride_d),
        offsets=(block_id*2,0),#block_id * block_size
        block_shape=(2,2),
        order=(1,0),
    )
    x = tl.load(block_x)
    x+=1
    tl.store(block_y, x)


x = torch.arange(32,dtype= torch.float32).reshape(8,4)[None,None,:,:].repeat(16,5,1,1).cuda()
print(x.shape)
y = torch.zeros((16, 5, 8, 4)).cuda()
add_one[(2,)](x, y, 5*8*4, 8*4, 4, 1)

print(torch.all(y == 0))