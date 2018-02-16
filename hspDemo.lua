require 'image'
require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'optim'
require 'cudnn'
dofile('utils/instanceNormalization.lua')

local hsp = require 'hsp'

cutorch.setDevice(arg[1])

local networkFileName = arg[2]
local imageFileName = arg[3]


local numLevels = 5
local splitThreshold = 0.08

local featBlockBoundary1, featBlockBoundary2, featBlockBoundary3

local featBlockBoundary1 = 12
local featBlockBoundary2 = 9
local featBlockBoundary3 = 20


-- load the snapshot
net = torch.load(networkFileName)
net:evaluate()


-- load the image
local img = image.load(imageFileName)
img = image.scale(img, 128, 128)
if (img:size()[1] == 4) then
  local alphaMask = img[4]:repeatTensor(3,1,1)
  img = torch.cmul(img:narrow(1,1,3),alphaMask) + 1 - alphaMask
end
img = img:view(1,3,128,128):cuda()


local volSoftMax = cudnn.VolumetricSoftMax():cuda()

-- forward pass through the network
outputs = {}
outputMasks = {}
outputs[1] = torch.Tensor(1,3,16,16,16):cuda():zero()
outputMasks[1] = torch.Tensor(1,1,16,16,16):cuda():zero()
for l=2,numLevels-1 do
  currSize = outputs[l-1]:size()
  outputs[l] = torch.Tensor(currSize[1],currSize[2],2*currSize[3],2*currSize[4],2*currSize[5]):cuda():zero()
  outputMasks[l] = torch.Tensor(currSize[1],1,2*currSize[3],2*currSize[4],2*currSize[5]):cuda():zero()
end
currSize = outputs[numLevels-1]:size()
outputs[numLevels] = torch.Tensor(currSize[1],1,2*currSize[3],2*currSize[4],2*currSize[5]):cuda():zero()
outputMasks[numLevels] = torch.Tensor(currSize[1],1,2*currSize[3],2*currSize[4],2*currSize[5]):cuda():zero()

-- run first stage 
result = net:get(1):forward(img)

local function evaluateFull(output, x, y, z, level)
  outputs[level][{{1,1},{},{16*x+1,16*(x+1)},{16*y+1,16*(y+1)},{16*z+1,16*(z+1)}}]:copy(output)
  outputMasks[level][{{1,1},{},{16*x+1,16*(x+1)},{16*y+1,16*(y+1)},{16*z+1,16*(z+1)}}]:fill(1)
end
      
local function evaluateIntermediate(output, feature, x, y, z, level)
  local outputSoftMax = volSoftMax:forward(output)

  -- copy output
  outputs[level][{{1,1},{},{16*x+1,16*(x+1)},{16*y+1,16*(y+1)},{16*z+1,16*(z+1)}}]:copy(output[{1,{1,3},{},{},{}}])
  outputMasks[level][{{1,1},{},{16*x+1,16*(x+1)},{16*y+1,16*(y+1)},{16*z+1,16*(z+1)}}]:fill(1)
  
  -- compute thresholds for splits
  local maxVal1 = torch.max(outputSoftMax[{1,3,{1,8},{1,8},{1,8}}])
  local maxVal2 = torch.max(outputSoftMax[{1,3,{1,8},{1,8},{9,16}}])
  local maxVal3 = torch.max(outputSoftMax[{1,3,{1,8},{9,16},{1,8}}])
  local maxVal4 = torch.max(outputSoftMax[{1,3,{1,8},{9,16},{9,16}}])
  local maxVal5 = torch.max(outputSoftMax[{1,3,{9,16},{1,8},{1,8}}])
  local maxVal6 = torch.max(outputSoftMax[{1,3,{9,16},{1,8},{9,16}}])
  local maxVal7 = torch.max(outputSoftMax[{1,3,{9,16},{9,16},{1,8}}])
  local maxVal8 = torch.max(outputSoftMax[{1,3,{9,16},{9,16},{9,16}}])
  
  if (maxVal1 > splitThreshold) then
    local result = net:get(level+1):forward(feature[{{1,1},{},{1,featBlockBoundary1},{1,featBlockBoundary1},{1,featBlockBoundary1}}])
    if (level < numLevels-1) then
      evaluateIntermediate(result[1],result[2], 2*x, 2*y, 2*z, level+1)
    else
      evaluateFull(result, 2*x, 2*y, 2*z, level+1)
    end
  end

  if (maxVal2 > splitThreshold) then
    local result = net:get(level+1):forward(feature[{{1,1},{},{1,featBlockBoundary1},{1,featBlockBoundary1},{featBlockBoundary2,featBlockBoundary3}}])
    if (level < numLevels-1) then
      evaluateIntermediate(result[1],result[2], 2*x, 2*y, 2*z+1, level+1)
    else
      evaluateFull(result,  2*x, 2*y, 2*z+1, level+1)
    end
  end
      
  if (maxVal3 > splitThreshold) then
    local result = net:get(level+1):forward(feature[{{1,1},{},{1,featBlockBoundary1},{featBlockBoundary2,featBlockBoundary3},{1,featBlockBoundary1}}])
    if (level < numLevels-1) then
      evaluateIntermediate(result[1],result[2], 2*x, 2*y+1, 2*z, level+1)
    else
      evaluateFull(result, 2*x, 2*y+1, 2*z, level+1)
    end
  end

  if (maxVal4 > splitThreshold) then
    local result = net:get(level+1):forward(feature[{{1,1},{},{1,featBlockBoundary1},{featBlockBoundary2,featBlockBoundary3},{featBlockBoundary2,featBlockBoundary3}}])
    if (level < numLevels-1) then
      evaluateIntermediate(result[1],result[2], 2*x, 2*y+1, 2*z+1, level+1)
    else
      evaluateFull(result, 2*x, 2*y+1, 2*z+1, level+1)
    end
  end

  if (maxVal5 > splitThreshold) then
    local result = net:get(level+1):forward(feature[{{1,1},{},{featBlockBoundary2,featBlockBoundary3},{1,featBlockBoundary1},{1,featBlockBoundary1}}])
    if (level < numLevels-1) then
      evaluateIntermediate(result[1],result[2], 2*x+1, 2*y, 2*z, level+1)
    else
      evaluateFull(result, 2*x+1, 2*y, 2*z, level+1)
    end
  end
  
  if (maxVal6 > splitThreshold) then
    local result = net:get(level+1):forward(feature[{{1,1},{},{featBlockBoundary2,featBlockBoundary3},{1,featBlockBoundary1},{featBlockBoundary2,featBlockBoundary3}}])
    if (level < numLevels-1) then
      evaluateIntermediate(result[1],result[2], 2*x+1, 2*y, 2*z+1, level+1)
    else
      evaluateFull(result, 2*x+1, 2*y, 2*z+1, level+1)
    end
  end

  if (maxVal7 > splitThreshold) then
    local result = net:get(level+1):forward(feature[{{1,1},{},{featBlockBoundary2,featBlockBoundary3},{featBlockBoundary2,featBlockBoundary3},{1,featBlockBoundary1}}])
    if (level < numLevels-1) then
      evaluateIntermediate(result[1],result[2], 2*x+1, 2*y+1, 2*z, level+1)
    else
      evaluateFull(result, 2*x+1, 2*y+1, 2*z, level+1)
    end
  end

  if (maxVal8 > splitThreshold) then
    local result = net:get(level+1):forward(feature[{{1,1},{},{featBlockBoundary2,featBlockBoundary3},{featBlockBoundary2,featBlockBoundary3},{featBlockBoundary2,featBlockBoundary3}}])
    if (level < numLevels-1) then
      evaluateIntermediate(result[1],result[2], 2*x+1, 2*y+1, 2*z+1, level+1)
    else
      evaluateFull(result, 2*x+1, 2*y+1, 2*z+1, level+1)
    end
  end
end
      
resultSingleOutput = result[1][{{1,1},{},{},{},{}}]
resultSingleFeature = result[2][{{1,1},{},{},{},{}}]
      
evaluateIntermediate(resultSingleOutput, resultSingleFeature, 0, 0, 0, 1)

local softMaxLayer = cudnn.VolumetricSoftMax():cuda()

outputsCompiled = {}

local upSampleLayer = nn.VolumetricFullConvolution(1,1,2,2,2,2,2,2,0,0,0)
upSampleLayer.bias:zero()
upSampleLayer.weight:fill(1)
upSampleLayer = upSampleLayer:cuda()

outputsCompiled = {}
outputsCompiled[1] = outputs[1]:clone()
local outputsCompiledColor = {}
for l=2,numLevels-1 do
  outputsCompiled[l] = outputs[l]:clone()
  
  local copyMask = torch.eq(outputMasks[l], 0):cuda()
      
  outputsCompiled[l][{{},{1},{},{},{}}]:add(upSampleLayer:forward(outputsCompiled[l-1][{{},{1},{},{},{}}]):cmul(copyMask))
  outputsCompiled[l][{{},{2},{},{},{}}]:add(upSampleLayer:forward(outputsCompiled[l-1][{{},{2},{},{},{}}]):cmul(copyMask))
  outputsCompiled[l][{{},{3},{},{},{}}]:add(upSampleLayer:forward(outputsCompiled[l-1][{{},{3},{},{},{}}]):cmul(copyMask))  
end

outputsCompiled[numLevels] = outputs[numLevels]:clone()
local copyMask = torch.eq(outputMasks[numLevels], 0):cuda()
local smOutputBefore = softMaxLayer:forward(outputsCompiled[numLevels-1])
local outputsCompiledTwoLabel = torch.add(smOutputBefore[{{},{2},{},{},{}}], smOutputBefore[{{},{3},{},{},{}}])
outputsCompiled[numLevels]:add(upSampleLayer:forward(outputsCompiledTwoLabel):cmul(copyMask))

hsp.saveMeshAsObj(outputsCompiled[numLevels][{1,1,{},{},{}}]:double(),0.2, "output.obj")