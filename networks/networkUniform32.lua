require 'cudnn'
dofile('utils/instanceNormalization.lua')


uniformNetwork = {}

function uniformNetwork.generateNewNetwork(nInputChannels)
  
  local net = nn.Sequential()
  
  -- self defined encoder
  net:add(cudnn.SpatialConvolution(nInputChannels, 16, 5, 5, 1, 1, 2, 2)) -- 128 x 128
  net:add(cudnn.SpatialMaxPooling(2,2,2,2))
  net:add(nn.SpatialInstanceNormalization(16))
  net:add(cudnn.ReLU(true))
  net:add(cudnn.SpatialConvolution(16, 32, 3, 3, 1, 1, 1, 1)) -- 64 x 64
  net:add(cudnn.SpatialMaxPooling(2,2,2,2))
  net:add(nn.SpatialInstanceNormalization(32))
  net:add(cudnn.ReLU(true))
  net:add(cudnn.SpatialConvolution(32, 64, 3, 3, 1, 1, 1, 1)) -- 32 x 32
  net:add(cudnn.SpatialMaxPooling(2,2,2,2))
  net:add(nn.SpatialInstanceNormalization(64))
  net:add(cudnn.ReLU(true))
  net:add(cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1)) -- 16 x 16
  net:add(cudnn.SpatialMaxPooling(2,2,2,2))
  net:add(nn.SpatialInstanceNormalization(128))
  net:add(cudnn.ReLU(true))
  net:add(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1)) -- 8 x 8
  net:add(cudnn.SpatialMaxPooling(2,2,2,2))
  net:add(nn.SpatialInstanceNormalization(256))
  net:add(cudnn.ReLU(true))
  net:add(cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1)) -- 4 x 4
  net:add(cudnn.SpatialMaxPooling(2,2,2,2))
  net:add(nn.SpatialInstanceNormalization(512))
  net:add(cudnn.ReLU(true))
  net:add(cudnn.SpatialConvolution(512,1024, 3, 3, 1, 1, 1, 1)) -- 2 x 2
  net:add(nn.SpatialInstanceNormalization(1024))
  net:add(nn.View(-1,2*2*1024))
  net:add(cudnn.ReLU(true))
  net:add(nn.Linear(2*2*1024,2*2*2*512))
  net:add(nn.View(-1,512,2,2,2))
  net:add(nn.VolumetricInstanceNormalization(512))
  net:add(cudnn.ReLU(true))
  
  net:add(cudnn.VolumetricFullConvolution(512, 256, 4, 4, 4, 2, 2, 2, 1, 1, 1)) --4x4x4
  net:add(nn.VolumetricInstanceNormalization(256))
  net:add(cudnn.ReLU(true))
  net:add(cudnn.VolumetricFullConvolution(256, 128, 4, 4, 4, 2, 2, 2, 1, 1, 1)) --8x8x8
  net:add(nn.VolumetricInstanceNormalization(128))
  net:add(cudnn.ReLU(true))
  net:add(cudnn.VolumetricFullConvolution(128, 128, 4, 4, 4, 2, 2, 2, 1, 1, 1)) --16x16x16
  net:add(cudnn.ReLU(true))
  net:add(cudnn.VolumetricConvolution(128, 64, 3, 3, 3, 1, 1, 1, 1, 1, 1)) --16x16x16
  net:add(cudnn.ReLU(true))
  net:add(cudnn.VolumetricFullConvolution(64, 64, 4, 4, 4, 2, 2, 2, 1, 1, 1)) --32x32x32
  net:add(cudnn.ReLU(true))
  net:add(cudnn.VolumetricConvolution(64, 1, 3, 3, 3, 1, 1, 1, 1, 1, 1)) --32x32x32
  net:add(cudnn.Sigmoid())

  return net
end


function uniformNetwork.saveSnapshot(i, snapshotFolder, net, optState)
  -- first we clear the state
  net:clearState()
  
  torch.save(snapshotFolder .. "/net" .. i ..".t7", net)
  
  if optState then
    optState.iter = i
    torch.save(snapshotFolder .. "/optState" .. i .. ".t7", optState)
  end
end

function uniformNetwork.loadSnapshot(snapshotFolder, iter, loadOptState)
  
  local net = torch.load(snapshotFolder .. "/net" .. iter .. ".t7")

  -- load opt state if there is one
  local optState
  if loadOptState then
    optState = torch.load(snapshotFolder .. "/optState" .. iter .. ".t7")
  end
  
  return net, optState
end
