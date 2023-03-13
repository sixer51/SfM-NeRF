import argparse
import glob
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter
import imageio
import torch

from DataLoader import *
from NeRFModel import *
from PositionEncoder import *
from render import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

def volumetricLoss(rgbMap, rgbMapGT):
    loss = torch.mean((rgbMap - rgbMapGT)**2)
    return loss

def render(rays, model, near, far, args, batchSize = 1024*16):
    rays_o, rays_d = rays
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    numRay = rays_o.shape[0]

    # viewdirs = rays_d
    # viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
    # viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    # shp = rays_d.shape
    # # Create ray batch
    # rays_o = torch.reshape(rays_o, [-1,3]).float()
    # rays_d = torch.reshape(rays_d, [-1,3]).float()
    
    # near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    # rays = torch.cat([rays_o, rays_d, near, far], -1)
    # rays = torch.cat([rays, viewdirs], -1)

    pts,viewdirs,z_vals = sample(rays_o,rays_d,near,far, args.n_sample)

    view_directions = viewdirs.view(numRay, 1, 3)
    view_directions = view_directions.expand([numRay, args.n_sample, 3])

    posEnc = positionEncoder(pts, args.n_pos_freq).reshape(numRay*args.n_sample, -1)
    dirEnc = positionEncoder(view_directions, args.n_dirc_freq).reshape(numRay*args.n_sample, -1)
    # raw = torch.cat([model(posEnc[i:i+batchSize], dirEnc[i:i+batchSize]) for i in range(0, posEnc.shape[0], batchSize)])

    outputs = []
    for i in range(0, posEnc.shape[0], batchSize):
        # torch.cuda.empty_cache()
        outputs.append(model(posEnc[i:i+batchSize], dirEnc[i:i+batchSize]))
    raw = torch.cat(outputs)

    raw = raw.reshape((-1, args.n_sample, 4))

    rgbMap = volumeRender(raw, z_vals, viewdirs)

    return rgbMap

def loadModel(model, args):
    startIter = 0
    files = glob.glob(args.checkpoint_path + '*.ckpt')
    latest_ckpt_file = max(files, key=os.path.getctime) if files else None

    if latest_ckpt_file and args.load_checkpoint:
        print(latest_ckpt_file)
        latest_ckpt = torch.load(latest_ckpt_file)
        startIter = latest_ckpt_file.replace(args.checkpoint_path,'').replace('model_','').replace('.ckpt','')
        startIter = int(startIter)
        model.load_state_dict(latest_ckpt['model_state_dict'])
        print(f"Loaded latest checkpoint from {latest_ckpt_file} ....")
    else:
        print('New model initialized....')
    
    return startIter

def train(images, poses, hwf, K, near, far, args):
    # setup tensorboard
    writer = SummaryWriter(args.logs_path)

    H, W, focal = hwf
    poses = torch.Tensor(poses).to(device)

    # setup encoder
    dimPosition = 3 * 2 * args.n_pos_freq + 3
    dimDirection = 3 * 2 * args.n_dirc_freq + 3

    # create model
    model = NeRFmodel(dimPos = dimPosition, dimDirc=dimDirection).to(device)
    # init optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lrate, betas=(0.9, 0.999))
    # load checkpoint
    startIter = loadModel(model, args)

    # start iteration
    numImg = len(images)
    for iter in tqdm(range(startIter, args.max_iters)):
        # choose random image
        imgID = random.randint(0, numImg-1)
        imageGT = images[imgID]
        imageGT = torch.Tensor(imageGT).to(device)
        poseGT = poses[imgID, :3,:4]

        numRay = args.n_rays_batch
        rays_o, rays_d = cameraFrameToRays(H, W, K, torch.Tensor(poseGT))
        coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
        coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
        
        # choose random rays
        select_inds = np.random.choice(coords.shape[0], size=[numRay], replace=False)  # (N_rand,)
        select_coords = coords[select_inds].long()  # (N_rand, 2)
        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        batch_rays = torch.stack([rays_o, rays_d], 0)
        rgbMapGT = imageGT[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        model.train()
        rgbMap = render(batch_rays, model, near, far, args)
        lossIter = volumetricLoss(rgbMap, rgbMapGT)

        optimizer.zero_grad()
        lossIter.backward()
        optimizer.step()
        
        if iter % 100 == 0:
            print(f"iter:{iter}, loss_this_iter:{lossIter}\n")
            writer.add_scalar('LossEveryIter', lossIter, iter)
            writer.flush()

        # save checkpoint
        if iter % args.save_ckpt_iter == 0:
            print("Saved a checkpoint {}".format(iter))
            if not (os.path.isdir(args.checkpoint_path)):
                os.makedirs(args.checkpoint_path)
            
            checkpoint_save_name =  args.checkpoint_path + os.sep + 'model_' + str(iter) + '.ckpt'
            torch.save({'iter': iter,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': lossIter}, checkpoint_save_name)

    print("Training is done")


def test(images, hwf, K, render_poses, near, far, args):
    render_poses = torch.Tensor(render_poses).to(device)
    H, W, focal = hwf

    dimPosition = 3 * 2 * args.n_pos_freq + 3
    dimDirection = 3 * 2 * args.n_dirc_freq + 3
    model = NeRFmodel(dimPos = dimPosition, dimDirc=dimDirection).to(device)
    loadModel(model, args)
    model.eval()

    if not (os.path.isdir(args.images_path)):
        os.makedirs(args.images_path)

    for i, pose in enumerate(render_poses):
        rays_o, rays_d = cameraFrameToRays(H, W, K, torch.Tensor(pose))
        batch_rays = torch.stack([rays_o, rays_d], 0)

        '''
        num_interval = 40
        interval = 400 // num_interval
        rgbMaps = []
        for i in range(num_interval):
            print(i)
            batch_rays_sub = batch_rays[:, :, interval*i:interval*(i+1), :]
            torch.cuda.empty_cache()
            rgbMap = render(batch_rays_sub, model, near, far, args)
            rgbMap = rgbMap.reshape((H, interval, 3))
            rgbMaps.append(rgbMap)
        image = torch.cat(rgbMaps, 1)
        '''
        rgbMap = render(batch_rays, model, near, far, args)
        image = rgbMap.reshape((H, W, 3))
        image = image.detach().cpu().numpy()
        cv2.imwrite("{}/view_{}.png".format(args.images_path, i), image)


def main(arg):
    # load data
    print("Loading data...")
    # images, poses, hwf, K, near, far, split = loadDataset("./SfM-Nerf/Phase2/data/lego/", testskip=args.testskip)
    images, poses, hwf, K, near, far, split = loadDataset("./Phase2/data/lego/", testskip=args.testskip)
    i_train, i_val, i_test = split
    images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])

    if arg.mode == 'train':
        print("Start training")
        train(images[i_train], poses[i_train], hwf, K, near, far, args)
    elif arg.mode == 'test':
        print("Start testing")
        render_poses = getRenderPose()
        test(images[i_test], hwf, K, render_poses, near, far, args)

def configParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--display',action='store_true',help="to display images")
    parser.add_argument('--dataset_path',default='../data/lego/',help="dataset path")
    parser.add_argument('--mode',default='test',help="train/test/val")
    parser.add_argument('--lrate',default=5e-4,help="training learning rate")
    parser.add_argument('--lrate_decay',default=25,help="decay learning rate")
    parser.add_argument('--n_ray_points',default=64,help="number of samples on a ray")
    parser.add_argument('--n_pos_freq',default=10,help="number of positional encoding frequencies for position")
    parser.add_argument('--n_dirc_freq',default=4,help="number of positional encoding frequencies for viewing direction")
    parser.add_argument('--n_rays_batch',default=32*32*4,help="number of rays per batch")
    parser.add_argument('--n_sample',default=32,help="number of rays per batch")
    parser.add_argument('--max_iters',default=10000,help="number of max iterations for training")
    parser.add_argument('--logs_path',default="./logs/",help="logs path")
    parser.add_argument('--checkpoint_path',default="./checkpoints/",help="checkpoints path")
    parser.add_argument('--load_checkpoint',default=False,help="whether to load checkpoint or not")
    parser.add_argument('--save_ckpt_iter',default=1000,help="num of iteration to save checkpoint")
    parser.add_argument('--images_path', default="./image/",help="folder to store images")
    parser.add_argument('--testskip', default=8,help="downsample test images")
    return parser

if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    main(args)