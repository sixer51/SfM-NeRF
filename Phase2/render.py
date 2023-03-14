import torch
import torch.nn.functional as F

def cameraFrameToRays(frameHeight, frameWidth, K, c2w):
    us, vs = torch.meshgrid(torch.linspace(0, frameWidth - 1, frameWidth), torch.linspace(0, frameHeight - 1, frameHeight))

    us = us.t().flatten()
    vs = vs.t().flatten()

    # xs = torch.stack((us,vs,torch.ones_like(us)))
    # rays_d = torch.matmul(torch.inverse(torch.tensor(K,dtype=torch.float32)), xs)
    rays_d = torch.stack([(us-K[0][2])/K[0][0], -(vs-K[1][2])/K[1][1], -torch.ones_like(us)], -1).transpose(0,1) #Not sure why it works this way...
    rays_d = torch.matmul(c2w[:3,:3], rays_d).transpose(0,1)

    rays_o = c2w[:3,-1].expand(rays_d.shape)

    return rays_o.reshape((frameWidth,frameHeight,3)), rays_d.reshape((frameWidth,frameHeight,3))


def sample(rays_o,rays_d,near_bound,far_bound,numSamples = 200):
    numRays = rays_d.shape[0]
    view_directions = rays_d/torch.norm(rays_d,dim=-1,keepdim=True)
    
    t = torch.linspace(0., 1., steps = numSamples)
    z = near_bound * (1. - t) + far_bound*t
    z = z.expand([numRays,numSamples])
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z[...,:,None]

    return pts, view_directions, z
    # return

# input model output
# output rgb image
def volumeRender(network_outputs, z_vals, rays_d):
    distances = z_vals[...,1:] - z_vals[...,:-1]
    distances = torch.cat([distances, torch.Tensor([1e10]).expand(distances[...,:1].shape)], -1)
    distances = distances*torch.norm(rays_d[...,None,:],dim=-1)
    rgb = torch.sigmoid(network_outputs[...,:3])

    alpha = 1. - torch.exp(-F.relu(network_outputs[...,3])*distances)
    weights = alpha*torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]

    weights_sum = torch.sum(weights,-1)
    rgb_final = torch.sum(weights[...,None]*rgb,-2)
    rgb_final = rgb_final + (1.-weights_sum[...,None])

    return rgb_final
