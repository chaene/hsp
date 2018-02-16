require 'cudnn'

local function addIntermediateUpsampleNetwork(subNet, numberOfChannels, padding, predColor)
  
  subNet:add(nn.Contiguous())
  if padding then
    subNet:add(cudnn.VolumetricFullConvolution(numberOfChannels[1], numberOfChannels[2], 4, 4, 4, 2, 2, 2, 2, 2, 2)) -- 22x22x22
  else 
    subNet:add(cudnn.VolumetricFullConvolution(numberOfChannels[1], numberOfChannels[2], 4, 4, 4, 2, 2, 2, 0, 0, 0)) -- 18x18x18
  end
  subNet:add(cudnn.ReLU(true))
  subNet:add(cudnn.VolumetricConvolution(numberOfChannels[2], numberOfChannels[3], 3, 3, 3, 1, 1, 1, 0, 0, 0))  -- 20x20x20 / 16x16x16
  subNet:add(cudnn.ReLU(true))
  
  
  local outputNet = nn.Sequential()
  if padding then
    outputNet:add(cudnn.VolumetricConvolution(numberOfChannels[3], numberOfChannels[4], 3, 3, 3, 1, 1, 1, 0, 0, 0)) -- 18x18x18
  else
    outputNet:add(cudnn.VolumetricConvolution(numberOfChannels[3], numberOfChannels[4], 3, 3, 3, 1, 1, 1, 1, 1, 1)) -- 16x16x16
  end
  outputNet:add(cudnn.ReLU(true))
  
  local numOutputChannels = 3
  if predColor then
    numOutputChannels = 6
  end
  
  if padding then
    outputNet:add(cudnn.VolumetricConvolution(numberOfChannels[4], numOutputChannels, 3, 3, 3, 1, 1, 1, 0, 0, 0)) -- 16x16x16
  else
    outputNet:add(cudnn.VolumetricConvolution(numberOfChannels[4], numOutputChannels, 3, 3, 3, 1, 1, 1, 1, 1, 1)) -- 16x16x16
  end
  
  
  local splitNet = nn.ConcatTable()
  splitNet:add(outputNet)
  splitNet:add(nn.Identity())
  
  subNet:add(splitNet)
  
end

hierarchicalSurfacePredictionNetwork = {}

function hierarchicalSurfacePredictionNetwork.generateNewNetwork(nInputChannels, numLevels, padding, predColor)
  
  local net = nn.Sequential()
  
  -- self defined encoder
  local stage1 = nn.Sequential()
  
  stage1:add(cudnn.SpatialConvolution(nInputChannels, 16, 5, 5, 1, 1, 2, 2)) -- 128 x 128
  stage1:add(cudnn.SpatialMaxPooling(2,2,2,2))
  stage1:add(nn.SpatialInstanceNormalization(16))
  stage1:add(cudnn.ReLU(true))
  stage1:add(cudnn.SpatialConvolution(16, 32, 3, 3, 1, 1, 1, 1)) -- 64 x 64
  stage1:add(cudnn.SpatialMaxPooling(2,2,2,2))
  stage1:add(nn.SpatialInstanceNormalization(32))
  stage1:add(cudnn.ReLU(true))
  stage1:add(cudnn.SpatialConvolution(32, 64, 3, 3, 1, 1, 1, 1)) -- 32 x 32
  stage1:add(cudnn.SpatialMaxPooling(2,2,2,2))
  stage1:add(nn.SpatialInstanceNormalization(64))
  stage1:add(cudnn.ReLU(true))
  stage1:add(cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1)) -- 16 x 16
  stage1:add(cudnn.SpatialMaxPooling(2,2,2,2))
  stage1:add(nn.SpatialInstanceNormalization(128))
  stage1:add(cudnn.ReLU(true))
  stage1:add(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1)) -- 8 x 8
  stage1:add(cudnn.SpatialMaxPooling(2,2,2,2))
  stage1:add(nn.SpatialInstanceNormalization(256))
  stage1:add(cudnn.ReLU(true))
  stage1:add(cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1)) -- 4 x 4
  stage1:add(cudnn.SpatialMaxPooling(2,2,2,2))
  stage1:add(nn.SpatialInstanceNormalization(512))
  stage1:add(cudnn.ReLU(true))
  stage1:add(cudnn.SpatialConvolution(512,1024, 3, 3, 1, 1, 1, 1)) -- 2 x 2
  stage1:add(nn.SpatialInstanceNormalization(1024))
  stage1:add(nn.View(-1,2*2*1024))
  stage1:add(cudnn.ReLU(true))
  if padding then
    stage1:add(nn.Linear(2*2*1024,3*3*3*512))
    stage1:add(nn.View(-1,512,3,3,3))
  else
    stage1:add(nn.Linear(2*2*1024,2*2*2*512))
    stage1:add(nn.View(-1,512,2,2,2))
  end
  stage1:add(nn.VolumetricInstanceNormalization(512))
  stage1:add(cudnn.ReLU(true))
  stage1:add(cudnn.VolumetricFullConvolution(512, 256, 4, 4, 4, 2, 2, 2, 1, 1, 1)) --6x6x6 / 4x4x4
  stage1:add(nn.VolumetricInstanceNormalization(256))
  stage1:add(cudnn.ReLU(true))
  stage1:add(cudnn.VolumetricFullConvolution(256, 128, 4, 4, 4, 2, 2, 2, 1, 1, 1)) --12x12x12 / 8x8x8
  stage1:add(nn.VolumetricInstanceNormalization(128))
  stage1:add(cudnn.ReLU(true))


  -- 8 -> 16
  addIntermediateUpsampleNetwork(stage1, {128, 128, 64, 32}, padding, predColor)
  --]]
  net:add(stage1)
  
  local fullInputChannels = 64

  if (numLevels > 2) then
    --16 -> 32
    local upsample1 = nn.Sequential()
    addIntermediateUpsampleNetwork(upsample1, {64, 64, 64, 32}, padding, predColor)
    net:add(upsample1)
    fullInputChannels = 64
  end
  
  if (numLevels > 3) then
    -- 32 -> 64
    local upsample2 = nn.Sequential()
    addIntermediateUpsampleNetwork(upsample2, {64, 64, 32, 16}, padding, predColor)
    net:add(upsample2)
    fullInputChannels = 32
  end
  
  if (numLevels > 4) then
    -- 64 -> 128
    local upsample3 = nn.Sequential()
    addIntermediateUpsampleNetwork(upsample3, {32, 32, 32, 16}, padding, predColor)
    net:add(upsample3)
    fullInputChannels = 32
  end

  -- 128 -> 256
  local fullNet = nn.Sequential()
  fullNet:add(nn.Contiguous())
  if padding then
    fullNet:add(cudnn.VolumetricFullConvolution(fullInputChannels, 16, 4, 4, 4, 2, 2, 2, 4, 4, 4)) -- 18x18x18
  else
    fullNet:add(cudnn.VolumetricFullConvolution(fullInputChannels, 16, 4, 4, 4, 2, 2, 2, 0, 0, 0)) -- 18x18x18
  end
  fullNet:add(cudnn.ReLU(true))
  if predColor then
    fullNet:add(cudnn.VolumetricConvolution(16, 4, 3, 3, 3, 1, 1, 1, 0, 0, 0)) -- 16x16x16
  else
    fullNet:add(cudnn.VolumetricConvolution(16, 1, 3, 3, 3, 1, 1, 1, 0, 0, 0)) -- 16x16x16
  end
  fullNet:add(cudnn.Sigmoid())
  
  net:add(fullNet)

  
  return net
end


function hierarchicalSurfacePredictionNetwork.saveSnapshot(i, snapshotFolder, net, optState)
  -- first we clear the state
  net:clearState()
  
  torch.save(snapshotFolder .. "/net" .. i ..".t7", net)
  
  if optState then
    optState.iter = i
    torch.save(snapshotFolder .. "/optState" .. i .. ".t7", optState)
  end
end

function hierarchicalSurfacePredictionNetwork.loadSnapshot(snapshotFolder, iter, loadOptState)
  
  local net = torch.load(snapshotFolder .. "/net" .. iter .. ".t7")

  -- load opt state if there is one
  local optState
  if loadOptState then
    optState = torch.load(snapshotFolder .. "/optState" .. iter .. ".t7")
  end
  
  return net, optState
end
