require('nn')
require('cudnn')
require('cutorch')
require('cunn')
require 'image'
require 'torch'
require 'optim'
require 'socket'
require 'lfs'

math.randomseed(os.time())

dofile('utils/instanceNormalization.lua')
dofile('networks/networkHierarchical.lua')


dofile('utils/hierarchicalShapenetDataLoader.lua')

local networkParameters = {}
local trainingParams = {}

local arg = arg

networkParameters, trainingParams = dofile(arg[2])
cutorch.setDevice(arg[1])

if (networkParameters.padding == nil) then
  error("networkParameters.padding needs to be set to true or false")
end

if (trainingParams.resolutionStr == nil) then
  error("trainingParams.resolutionStr as to be set")
end

local predColor = networkParameters.predictColors

local startIteration = 1
local net
local optimState = {}

local numLevels = networkParameters.numLevels
if (numLevels ~= 5 and numLevels ~= 2) then
  error("This code is for a 5 or 2 level tree")
end

local padding = networkParameters.padding

if not trainingParams.resumeIteration then
  if (networkParameters.useColor) then
    net = hierarchicalSurfacePredictionNetwork.generateNewNetwork(3, numLevels, padding, predColor):cuda()
  else
    net = hierarchicalSurfacePredictionNetwork.generateNewNetwork(1, numLevels, padding, predColor):cuda()
  end
else
  net, optimState =hierarchicalSurfacePredictionNetwork.loadSnapshot(trainingParams.outputFolder .. "/snapshot/", trainingParams.resumeIteration, true)
  startIteration = optimState.iter + 1
  net:training()
end

local levelTakeProbs = trainingParams.levelTakeProbs

if (numLevels == 5) then
  print("Level Take Probabilities: {" .. levelTakeProbs[1] .. ", " .. levelTakeProbs[2] .. ", " .. levelTakeProbs[3] .. ", " .. levelTakeProbs[4] .. "}")
elseif (numLevels == 2) then
  print("Level Take Probabilities: {" .. levelTakeProbs[1] .. "}")
end

valLogger = optim.Logger(trainingParams.outputFolder .. 'training_valLoss_' .. startIteration .. '.log')
if (numLevels == 5) then
  if predColor then
    valLogger:setNames{'Iteration', 'L1', 'L2', 'L3', 'L4', 'L5', 'LC1', 'LC2', 'LC3', 'LC4', 'LC5', 'time'}
  else
    valLogger:setNames{'Iteration', 'L1', 'L2', 'L3', 'L4', 'L5', 'time'}
  end
  
elseif (numLevels == 2) then
  valLogger:setNames{'Iteration', 'L1', 'L2', 'time'}
end

trainLogger = optim.Logger(trainingParams.outputFolder .. 'training_trainLoss_' .. startIteration .. '.log')
if (numLevels == 5) then
  if predColor then
    trainLogger:setNames{'Iteration', 'L1', 'L2', 'L3', 'L4', 'L5', 'LC1', 'LC2', 'LC3', 'LC4', 'LC5', 'time'}
  else
    trainLogger:setNames{'Iteration', 'L1', 'L2', 'L3', 'L4', 'L5', 'time'}
  end
elseif (numLevels == 2) then
  trainLogger:setNames{'Iteration', 'L1', 'L2', 'time'}
end

local splitThreshold = networkParameters.splitThreshold
local optimParams = trainingParams.config
local usePredSplit = trainingParams.usePredictedSplit

-- set up the data set
hierarchical3DShapeNetDataSet:setResolution(trainingParams.resolutionStr)

if networkParameters.useColor then
  hierarchical3DShapeNetDataSet:setColor()
end

if predColor then
  hierarchical3DShapeNetDataSet:setLoadVoxelColors()
end


hierarchical3DShapeNetDataSet:setPath(trainingParams.datasetFolder)
for c=1,#trainingParams.valClasses do
  hierarchical3DShapeNetDataSet:addValDataFromClass(trainingParams.valClasses[c])
end

for c=1,#trainingParams.trainClasses do
  hierarchical3DShapeNetDataSet:addTrainDataFromClass(trainingParams.trainClasses[c])
end

hierarchical3DShapeNetDataSet:shuffleTrainDataFileNames()

-- generate snapshot folder
local snapshotFolderExists = lfs.attributes(trainingParams.outputFolder .. "/snapshot/",'modification')
if (snapshotFolderExists == nil) then
  lfs.mkdir(trainingParams.outputFolder .. "/snapshot/")
end


-- sliding window training loss
local trainingLossBufferSize = 200
local trainingLossBuffer = torch.Tensor(trainingLossBufferSize, numLevels):zero()
local trainingLossBufferPointer = 1

local trainingLossBufferColors
if predColor then
 trainingLossBufferColors = torch.Tensor(trainingLossBufferSize, numLevels):zero() 
end

local totalTrainingTime = 0

cutorch.synchronize()
local start = socket.gettime()

-- layer to compute volumetric soft max
local volSoftMax = cudnn.VolumetricSoftMax():cuda()

local colorWeight = 10

-- loss layers
local bceCrit = nn.BCECriterion():cuda()
local mseCrit = nn.AbsCriterion():cuda()
-- having one criterion per level avoids copying the gradients
-- TODO: figure out if the additional table overhead is faster than copying
local volCECrits = {}
for l=1,numLevels-1 do
  volCECrits[l] = cudnn.VolumetricCrossEntropyCriterion():cuda()
end


-- compute loss weights
local lossWeights = torch.Tensor(numLevels)
local lossWeightsColor = torch.Tensor(numLevels)
local curLossWeight = 1/trainingParams.batchSize
local curLossWeightColor = 1/trainingParams.batchSize
for l=1,numLevels-1 do
  lossWeights[l] = curLossWeight
  lossWeightsColor[l] = curLossWeightColor
  curLossWeight = curLossWeight/(trainingParams.levelLossDecay*levelTakeProbs[l])
  curLossWeightColor = 2*curLossWeightColor/(trainingParams.levelLossDecay*levelTakeProbs[l]) -- compensate for surface only ground truth
end


lossWeights[numLevels] = curLossWeight
lossWeightsColor[numLevels] = curLossWeightColor

local lossLevels = torch.Tensor(numLevels):zero()
local colorLossLevels = torch.Tensor(numLevels):zero()

-- allocate structures for the input to the network
local batchInput
if (networkParameters.useColor) then
  batchInput = torch.Tensor(trainingParams.batchSize, 3, 128, 128):cuda()
else
  batchInput = torch.Tensor(trainingParams.batchSize, 1, 128, 128):cuda()
end
batchBlockindices = {}
batchBlocks = {}
for b=1,trainingParams.batchSize do
  batchBlocks[b] = {}
end
local batchVoxelColors = {}
for b=1,trainingParams.batchSize do
  batchVoxelColors[b] = {}
  for l=1,numLevels do
    batchVoxelColors[b][l] = {}
  end
end

-- allocate structures for gradients
local gradFeatures 
if padding then
  gradFeatures = torch.Tensor(numLevels, trainingParams.batchSize, 64, 20, 20, 20):cuda()
else
  gradFeatures = torch.Tensor(numLevels, trainingParams.batchSize, 64, 16, 16, 16):cuda()
end
local gradOutput
if predColor then
  gradOutput = torch.Tensor(1, trainingParams.batchSize, 6, 16, 16, 16):cuda()
else
  gradOutput = torch.Tensor(1, trainingParams.batchSize, 3, 16, 16, 16):cuda()
end
local blockIndices
local blocks
local voxelColors

local colorBlockGT
local colorBlockMask
local colorBlockMaskInv

if predColor then
  colorBlockGT = torch.Tensor(3, 16, 16, 16):cuda()
  colorBlockMask = torch.Tensor(3, 16, 16, 16):zero():cuda()
  colorBlockMaskInv = torch.Tensor(3, 16, 16, 16):zero():cuda()
end

local gradJoingOutput
if predColor then
  gradJointOutput = torch.Tensor(numLevels, 1, 6,16,16,16):cuda()
end

local featBlockBoundary1, featBlockBoundary2, featBlockBoundary3
if padding then
  featBlockBoundary1 = 12
  featBlockBoundary2 = 9
  featBlockBoundary3 = 20
else
  featBlockBoundary1 = 8
  featBlockBoundary2 = 9
  featBlockBoundary3 = 16
end

local params, gradParams = net:getParameters()

for i = startIteration, 1000000 do
  
  local function evaluateGradientFull(output, x, y, z, level)
    
    if predColor then
      
      local labelIndex = blockIndices[level][x+1][y+1][z+1]
      local colorIndex = voxelColors[level].bi[x+1][y+1][z+1]
      
      colorBlockMask[{1,{},{},{}}] = voxelColors[level].bm[colorIndex]
      colorBlockMask[{2,{},{},{}}] = voxelColors[level].bm[colorIndex]
      colorBlockMask[{3,{},{},{}}] = voxelColors[level].bm[colorIndex]
      
      colorBlockMaskInv:copy(colorBlockMask)
      colorBlockMaskInv:csub(1)
      colorBlockMaskInv:mul(-1)
      
      colorBlockGT[{{},{},{},{}}] = output[{1,{2,4},{},{},{}}] --copy solution to GT
      colorBlockGT:cmul(colorBlockMaskInv)
      colorBlockGT[{{},{},{},{}}] = colorBlockGT[{{1,3},{},{},{}}] + voxelColors[level].bc[colorIndex]
      
      
      local lossOcc = bceCrit:forward(output[{1,1,{},{},{}}], blocks[level][labelIndex])*lossWeights[level]
      local lossColor = mseCrit:forward(output[{1,{2,4},{},{},{}}], colorBlockGT)*lossWeightsColor[level]*colorWeight
      
      lossLevels[level] = lossLevels[level] + lossOcc
      colorLossLevels[level] = colorLossLevels[level] + lossColor
      
      gradJointOutput[level][{1,1,{},{},{}}]:copy(bceCrit:backward(output[{1,1,{},{},{}}], blocks[level][labelIndex]):mul(lossWeights[level]))
      gradJointOutput[level][{1,{2,4},{},{},{}}]:copy(mseCrit:backward(output[{1,{2,4},{},{},{}}], colorBlockGT):cmul(colorBlockMask):mul(lossWeightsColor[level]*colorWeight))
      return gradJointOutput[level]
      
    else
      local labelIndex = blockIndices[level][x+1][y+1][z+1]
      local loss = bceCrit:forward(output, blocks[level][labelIndex])*lossWeights[level]
      lossLevels[level] = lossLevels[level] + loss
      local curGradOutput = bceCrit:backward(output, blocks[level][labelIndex]):mul(lossWeights[level])
      return curGradOutput
      
    end
  end
    
  local function evaluateGradientIntermediate(output, feature, x, y, z, level)
    local outputSoftMax = volSoftMax:forward(output[{{1,1},{1,3},{},{},{}}])
    
    local nChannels = feature:size()[2]
    
    local labelIndex = blockIndices[level][x+1][y+1][z+1]
    --TODO: store blocks such that dimensions are already matching
    local lossOcc = volCECrits[level]:forward(output[{{1,1},{1,3},{},{},{}}], blocks[level][labelIndex]:view(1,16,16,16))*lossWeights[level]
    
    if predColor then
      
      local colorIndex = voxelColors[level].bi[x+1][y+1][z+1]
      
      colorBlockMask[{1,{},{},{}}] = voxelColors[level].bm[colorIndex]
      colorBlockMask[{2,{},{},{}}] = voxelColors[level].bm[colorIndex]
      colorBlockMask[{3,{},{},{}}] = voxelColors[level].bm[colorIndex]
      
      colorBlockMaskInv:copy(colorBlockMask)
      colorBlockMaskInv:csub(1)
      colorBlockMaskInv:mul(-1)
      
      colorBlockGT[{{},{},{},{}}] = output[{1,{4,6},{},{},{}}] --copy solution to GT
      colorBlockGT:cmul(colorBlockMaskInv)
      colorBlockGT[{{},{},{},{}}] = colorBlockGT[{{1,3},{},{},{}}] + voxelColors[level].bc[colorIndex]
    
      local lossColor = mseCrit:forward(output[{1,{4,6},{},{},{}}], colorBlockGT)*lossWeightsColor[level]*colorWeight
      
      colorLossLevels[level] = colorLossLevels[level] + lossColor
    end
    
    lossLevels[level] = lossLevels[level] + lossOcc
    local curGradOutput
    if predColor then
      gradJointOutput[level][{1,{1,3},{},{},{}}]:copy(volCECrits[level]:backward(output[{{1,1},{1,3},{},{},{}}], blocks[level][labelIndex]:view(1,16,16,16)):mul(lossWeights[level]))
      gradJointOutput[level][{1,{4,6},{},{},{}}]:copy(mseCrit:backward(output[{1,{4,6},{},{},{}}], colorBlockGT):cmul(colorBlockMask):mul(lossWeightsColor[level]*colorWeight))
      curGradOutput = gradJointOutput[level]
    else
      curGradOutput = volCECrits[level]:backward(output, blocks[level][labelIndex]:view(1,16,16,16)):mul(lossWeights[level])
    end
    
    -- zero the grad inputs
    gradFeatures[level+1][1]:zero()
    
    if (level < trainingParams.stopAtLevel) then
    
    -- to avoid the usage of additional tables and loops
    -- the code for each octant is explicitly written
    
    -- compute thresholds for splits
    local maxVal1 = 0
    local maxVal2 = 0
    local maxVal3 = 0
    local maxVal4 = 0
    local maxVal5 = 0
    local maxVal6 = 0
    local maxVal7 = 0
    local maxVal8 = 0
    if (usePredSplit) then
      maxVal1 = torch.max(outputSoftMax[{1,3,{1,8},{1,8},{1,8}}])
      maxVal2 = torch.max(outputSoftMax[{1,3,{1,8},{1,8},{9,16}}])
      maxVal3 = torch.max(outputSoftMax[{1,3,{1,8},{9,16},{1,8}}])
      maxVal4 = torch.max(outputSoftMax[{1,3,{1,8},{9,16},{9,16}}])
      maxVal5 = torch.max(outputSoftMax[{1,3,{9,16},{1,8},{1,8}}])
      maxVal6 = torch.max(outputSoftMax[{1,3,{9,16},{1,8},{9,16}}])
      maxVal7 = torch.max(outputSoftMax[{1,3,{9,16},{9,16},{1,8}}])
      maxVal8 = torch.max(outputSoftMax[{1,3,{9,16},{9,16},{9,16}}])
    end
    
    local maxValGT = torch.max(blocks[level][labelIndex][{{1,8},{1,8},{1,8}}])
    if ((maxVal1 > splitThreshold or maxValGT > 2.5) and (math.random(100000) <= levelTakeProbs[level]*100000)) then
      local result = net:get(level+1):forward(feature[{{1,1},{},{1,featBlockBoundary1},{1,featBlockBoundary1},{1,featBlockBoundary1}}])
      if (level < numLevels-1) then
	local nextGradOutput, nextGradFeatures = evaluateGradientIntermediate(result[1],result[2], 2*x, 2*y, 2*z, level+1)
	gradFeatures[level+1][{{1,1},{1,nChannels},{1,featBlockBoundary1},{1,featBlockBoundary1},{1,featBlockBoundary1}}]:add(net:get(level+1):backward(feature[{{1,1},{},{1,featBlockBoundary1},{1,featBlockBoundary1},{1,featBlockBoundary1}}],{nextGradOutput, nextGradFeatures}))
      else
	local nextGradOutput = evaluateGradientFull(result, 2*x, 2*y, 2*z, level+1)
	gradFeatures[level+1][{{1,1},{1,nChannels},{1,featBlockBoundary1},{1,featBlockBoundary1},{1,featBlockBoundary1}}]:add(net:get(level+1):backward(feature[{{1,1},{},{1,featBlockBoundary1},{1,featBlockBoundary1},{1,featBlockBoundary1}}], nextGradOutput))
      end
    end
    
    maxValGT = torch.max(blocks[level][labelIndex][{{1,8},{1,8},{9,16}}])
    if ((maxVal2 > splitThreshold or maxValGT > 2.5) and (math.random(100000) <= levelTakeProbs[level]*100000)) then
      local result = net:get(level+1):forward(feature[{{1,1},{},{1,featBlockBoundary1},{1,featBlockBoundary1},{featBlockBoundary2,featBlockBoundary3}}])
      if (level < numLevels-1) then
	local nextGradOutput, nextGradFeatures = evaluateGradientIntermediate(result[1],result[2], 2*x, 2*y, 2*z+1, level+1)
	gradFeatures[level+1][{{1,1},{1,nChannels},{1,featBlockBoundary1},{1,featBlockBoundary1},{featBlockBoundary2,featBlockBoundary3}}]:add(net:get(level+1):backward(feature[{{1,1},{},{1,featBlockBoundary1},{1,featBlockBoundary1},{featBlockBoundary2,featBlockBoundary3}}],{nextGradOutput, nextGradFeatures}))
      else
	local nextGradOutput = evaluateGradientFull(result, 2*x, 2*y, 2*z+1, level+1)
	gradFeatures[level+1][{{1,1},{1,nChannels},{1,featBlockBoundary1},{1,featBlockBoundary1},{featBlockBoundary2,featBlockBoundary3}}]:add(net:get(level+1):backward(feature[{{1,1},{},{1,featBlockBoundary1},{1,featBlockBoundary1},{featBlockBoundary2,featBlockBoundary3}}], nextGradOutput))
      end
    end
    
    maxValGT = torch.max(blocks[level][labelIndex][{{1,8},{9,16},{1,8}}])
    if ((maxVal3 > splitThreshold or maxValGT > 2.5) and (math.random(100000) <= levelTakeProbs[level]*100000)) then
      local result = net:get(level+1):forward(feature[{{1,1},{},{1,featBlockBoundary1},{featBlockBoundary2,featBlockBoundary3},{1,featBlockBoundary1}}])
      if (level < numLevels-1) then
	local nextGradOutput, nextGradFeatures = evaluateGradientIntermediate(result[1],result[2], 2*x, 2*y+1, 2*z, level+1)
	gradFeatures[level+1][{{1,1},{1,nChannels},{1,featBlockBoundary1},{featBlockBoundary2,featBlockBoundary3},{1,featBlockBoundary1}}]:add(net:get(level+1):backward(feature[{{1,1},{},{1,featBlockBoundary1},{featBlockBoundary2,featBlockBoundary3},{1,featBlockBoundary1}}],{nextGradOutput, nextGradFeatures}))
      else
	local nextGradOutput = evaluateGradientFull(result, 2*x, 2*y+1, 2*z, level+1)
	gradFeatures[level+1][{{1,1},{1,nChannels},{1,featBlockBoundary1},{featBlockBoundary2,featBlockBoundary3},{1,featBlockBoundary1}}]:add(net:get(level+1):backward(feature[{{1,1},{},{1,featBlockBoundary1},{featBlockBoundary2,featBlockBoundary3},{1,featBlockBoundary1}}], nextGradOutput))
      end
    end
    
    maxValGT = torch.max(blocks[level][labelIndex][{{1,8},{9,16},{9,16}}])
    if ((maxVal4 > splitThreshold or maxValGT > 2.5) and (math.random(100000) <= levelTakeProbs[level]*100000)) then
      local result = net:get(level+1):forward(feature[{{1,1},{},{1,featBlockBoundary1},{featBlockBoundary2,featBlockBoundary3},{featBlockBoundary2,featBlockBoundary3}}])
      if (level < numLevels-1) then
	local nextGradOutput, nextGradFeatures = evaluateGradientIntermediate(result[1],result[2], 2*x, 2*y+1, 2*z+1, level+1)
	gradFeatures[level+1][{{1,1},{1,nChannels},{1,featBlockBoundary1},{featBlockBoundary2,featBlockBoundary3},{featBlockBoundary2,featBlockBoundary3}}]:add(net:get(level+1):backward(feature[{{1,1},{},{1,featBlockBoundary1},{featBlockBoundary2,featBlockBoundary3},{featBlockBoundary2,featBlockBoundary3}}],{nextGradOutput, nextGradFeatures}))
      else
	local nextGradOutput = evaluateGradientFull(result, 2*x, 2*y+1, 2*z+1, level+1)
	gradFeatures[level+1][{{1,1},{1,nChannels},{1,featBlockBoundary1},{featBlockBoundary2,featBlockBoundary3},{featBlockBoundary2,featBlockBoundary3}}]:add(net:get(level+1):backward(feature[{{1,1},{},{1,featBlockBoundary1},{featBlockBoundary2,featBlockBoundary3},{featBlockBoundary2,featBlockBoundary3}}], nextGradOutput))
      end
    end
    
    maxValGT = torch.max(blocks[level][labelIndex][{{9,16},{1,8},{1,8}}])
    if ((maxVal5 > splitThreshold or maxValGT > 2.5) and (math.random(100000) <= levelTakeProbs[level]*100000)) then
      local result = net:get(level+1):forward(feature[{{1,1},{},{featBlockBoundary2,featBlockBoundary3},{1,featBlockBoundary1},{1,featBlockBoundary1}}])
      if (level < numLevels-1) then
	local nextGradOutput, nextGradFeatures = evaluateGradientIntermediate(result[1],result[2], 2*x+1, 2*y, 2*z, level+1)
	gradFeatures[level+1][{{1,1},{1,nChannels},{featBlockBoundary2,featBlockBoundary3},{1,featBlockBoundary1},{1,featBlockBoundary1}}]:add(net:get(level+1):backward(feature[{{1,1},{},{featBlockBoundary2,featBlockBoundary3},{1,featBlockBoundary1},{1,featBlockBoundary1}}],{nextGradOutput, nextGradFeatures}))
      else
	local nextGradOutput = evaluateGradientFull(result, 2*x+1, 2*y, 2*z, level+1)
	gradFeatures[level+1][{{1,1},{1,nChannels},{featBlockBoundary2,featBlockBoundary3},{1,featBlockBoundary1},{1,featBlockBoundary1}}]:add(net:get(level+1):backward(feature[{{1,1},{},{featBlockBoundary2,featBlockBoundary3},{1,featBlockBoundary1},{1,featBlockBoundary1}}], nextGradOutput))
      end
    end
        
    maxValGT = torch.max(blocks[level][labelIndex][{{9,16},{1,8},{9,16}}])
    if ((maxVal6 > splitThreshold or maxValGT > 2.5) and (math.random(100000) <= levelTakeProbs[level]*100000)) then
      local result = net:get(level+1):forward(feature[{{1,1},{},{featBlockBoundary2,featBlockBoundary3},{1,featBlockBoundary1},{featBlockBoundary2,featBlockBoundary3}}])
      if (level < numLevels-1) then
	local nextGradOutput, nextGradFeatures = evaluateGradientIntermediate(result[1],result[2], 2*x+1, 2*y, 2*z+1, level+1)
	gradFeatures[level+1][{{1,1},{1,nChannels},{featBlockBoundary2,featBlockBoundary3},{1,featBlockBoundary1},{featBlockBoundary2,featBlockBoundary3}}]:add(net:get(level+1):backward(feature[{{1,1},{},{featBlockBoundary2,featBlockBoundary3},{1,featBlockBoundary1},{featBlockBoundary2,featBlockBoundary3}}],{nextGradOutput, nextGradFeatures}))
      else
	local nextGradOutput = evaluateGradientFull(result, 2*x+1, 2*y, 2*z+1, level+1)
	gradFeatures[level+1][{{1,1},{1,nChannels},{featBlockBoundary2,featBlockBoundary3},{1,featBlockBoundary1},{featBlockBoundary2,featBlockBoundary3}}]:add(net:get(level+1):backward(feature[{{1,1},{},{featBlockBoundary2,featBlockBoundary3},{1,featBlockBoundary1},{featBlockBoundary2,featBlockBoundary3}}], nextGradOutput))
      end
    end
    
    maxValGT = torch.max(blocks[level][labelIndex][{{9,16},{9,16},{1,8}}])
    if ((maxVal7 > splitThreshold or maxValGT > 2.5) and (math.random(100000) <= levelTakeProbs[level]*100000)) then
      local result = net:get(level+1):forward(feature[{{1,1},{},{featBlockBoundary2,featBlockBoundary3},{featBlockBoundary2,featBlockBoundary3},{1,featBlockBoundary1}}])
      if (level < numLevels-1) then
	local nextGradOutput, nextGradFeatures = evaluateGradientIntermediate(result[1],result[2], 2*x+1, 2*y+1, 2*z, level+1)
	gradFeatures[level+1][{{1,1},{1,nChannels},{featBlockBoundary2,featBlockBoundary3},{featBlockBoundary2,featBlockBoundary3},{1,featBlockBoundary1}}]:add(net:get(level+1):backward(feature[{{1,1},{},{featBlockBoundary2,featBlockBoundary3},{featBlockBoundary2,featBlockBoundary3},{1,featBlockBoundary1}}],{nextGradOutput, nextGradFeatures}))
      else
	local nextGradOutput = evaluateGradientFull(result, 2*x+1, 2*y+1, 2*z, level+1)
	gradFeatures[level+1][{{1,1},{1,nChannels},{featBlockBoundary2,featBlockBoundary3},{featBlockBoundary2,featBlockBoundary3},{1,featBlockBoundary1}}]:add(net:get(level+1):backward(feature[{{1,1},{},{featBlockBoundary2,featBlockBoundary3},{featBlockBoundary2,featBlockBoundary3},{1,featBlockBoundary1}}], nextGradOutput))
      end
    end
        
    maxValGT = torch.max(blocks[level][labelIndex][{{9,16},{9,16},{9,16}}])
    if ((maxVal8 > splitThreshold or maxValGT > 2.5) and (math.random(100000) <= levelTakeProbs[level]*100000)) then
      local result = net:get(level+1):forward(feature[{{1,1},{},{featBlockBoundary2,featBlockBoundary3},{featBlockBoundary2,featBlockBoundary3},{featBlockBoundary2,featBlockBoundary3}}])
      if (level < numLevels-1) then
	local nextGradOutput, nextGradFeatures = evaluateGradientIntermediate(result[1],result[2], 2*x+1, 2*y+1, 2*z+1, level+1)
	gradFeatures[level+1][{{1,1},{1,nChannels},{featBlockBoundary2,featBlockBoundary3},{featBlockBoundary2,featBlockBoundary3},{featBlockBoundary2,featBlockBoundary3}}]:add(net:get(level+1):backward(feature[{{1,1},{},{featBlockBoundary2,featBlockBoundary3},{featBlockBoundary2,featBlockBoundary3},{featBlockBoundary2,featBlockBoundary3}}],{nextGradOutput, nextGradFeatures}))
      else
	local nextGradOutput = evaluateGradientFull(result, 2*x+1, 2*y+1, 2*z+1, level+1)
	gradFeatures[level+1][{{1,1},{1,nChannels},{featBlockBoundary2,featBlockBoundary3},{featBlockBoundary2,featBlockBoundary3},{featBlockBoundary2,featBlockBoundary3}}]:add(net:get(level+1):backward(feature[{{1,1},{},{featBlockBoundary2,featBlockBoundary3},{featBlockBoundary2,featBlockBoundary3},{featBlockBoundary2,featBlockBoundary3}}], nextGradOutput))
      end
    end
  
    end
    return curGradOutput, gradFeatures[level+1][1][{{1,nChannels}}]
  end
  
  function evalGradients(params)
    gradParams:zero()
    
    
    -- run first stage 
    result = net:get(1):forward(batchInput)
    
    for b=1,trainingParams.batchSize do
      
      singleOutput = result[1][{{b,b},{},{},{},{}}]
      singleFeature = result[2][{{b,b},{},{},{},{}}]
      
      
      blockIndices = batchBlockindices[b]
      blocks = batchBlocks[b]
      if predColor then
	voxelColors = batchVoxelColors[b]
      end
      local gradOutputSingle, gradFeaturesSingle = evaluateGradientIntermediate(singleOutput, singleFeature, 0, 0, 0, 1)
      
      gradFeatures[1][{{b,b},{},{},{},{}}]:copy(gradFeaturesSingle)
      gradOutput[1][{{b,b},{},{},{},{}}]:copy(gradOutputSingle)
    end
    
    net:get(1):backward(batchInput,{gradOutput[1], gradFeatures[1]})
    
    
    -- return 0 as loss
    return 0, gradParams
  end
  
  -- load a new batch
  for b=1,trainingParams.batchSize do
    local obs, blockIndices, blocks, modelName, curVoxelColors = hierarchical3DShapeNetDataSet:getNextTrainExample(false, true)
    batchInput[{b,{},{},{}}] = obs:cuda()
    batchBlockindices[b] = blockIndices
    for l=1,numLevels do
      batchBlocks[b][l] = blocks[l]:cuda()
    end
    if predColor then
      for l=1,numLevels do
        batchVoxelColors[b][l].bi = curVoxelColors[l].bi:cuda()
        batchVoxelColors[b][l].bm = curVoxelColors[l].bm:cuda()
        batchVoxelColors[b][l].bc = curVoxelColors[l].bc:cuda()
      end
    end
  end
  
  -- zero buffers
  lossLevels:zero()
  if predColor then
    colorLossLevels:zero()
  end
  gradOutput:zero()
  gradFeatures:zero()
  
  -- do the optimization step
  optim.sgd(evalGradients, params, optimParams, optimState)
  
  for l=1,numLevels do
    trainingLossBuffer[trainingLossBufferPointer][l] = lossLevels[l]
    if predColor then
      trainingLossBufferColors[trainingLossBufferPointer][l] = colorLossLevels[l]
    end
  end
  
  trainingLossBufferPointer = trainingLossBufferPointer + 1
  if (trainingLossBufferPointer > trainingLossBufferSize) then
    trainingLossBufferPointer = 1
  end
   
  -- output loss every 5 iterations
  if i % 5 == 0 then
    local losses = torch.sum(trainingLossBuffer, 1)/math.min(trainingLossBufferSize,i-startIteration+1)

    print("iteration: " .. i .. ", sliding window training loss : ")
    print("geometry loss")
    print(losses)
    
    local colorLosses
    if predColor then
      colorLosses = torch.sum(trainingLossBufferColors, 1)/math.min(trainingLossBufferSize,i-startIteration+1)
      print("color loss")
      print(colorLosses)
    end
    print("")
    
    local curEnd = socket.gettime()
      
    if (numLevels == 5) then
      if predColor then
        trainLogger:add{i,losses[1][1], losses[1][2], losses[1][3], losses[1][4], losses[1][5], 
                    colorLosses[1][1], colorLosses[1][2], colorLosses[1][3], colorLosses[1][4], colorLosses[1][5], curEnd - start}
      else
        trainLogger:add{i,losses[1][1], losses[1][2], losses[1][3], losses[1][4], losses[1][5], curEnd - start}
      end
    elseif (numLevels == 2) then
      trainLogger:add{i,losses[1][1], losses[1][2], curEnd - start}
    end
  end
  
  if (i % trainingParams.validationInterval == 0) then
    hierarchicalSurfacePredictionNetwork.saveSnapshot(i, trainingParams.outputFolder .. "/snapshot/", net, optimState)
    
    net:evaluate()
    
    local valBatchSize = 1
    
    local valLosses = torch.Tensor(numLevels):zero()
    local valLossesColor = torch.Tensor(numLevels):zero()
    local numValidatedElements = 0
    
    for j=1,#hierarchical3DShapeNetDataSet.valDataFiles do
      if ((j-1)%trainingParams.validationSubsampling == 0) then
        local obs, blockIndices, blocks, modelName, voxelColors = hierarchical3DShapeNetDataSet:getNextValExample(false, false)
        
	if networkParameters.useColor then
	  input = obs:cuda():view(1,3,128,128)
	else
	  input = obs:cuda():view(1,1,128,128)
	end
        
        for l=1,numLevels do
          blocks[l] = blocks[l]:cuda()
        end
        
        print("Running validation model " .. j .. " : " .. modelName)
        
        -- forward pass through the network
        outputs = {}
        outputsColor = {}
        outputMasks = {}
        outputs[1] = torch.Tensor(valBatchSize,3,16,16,16):cuda():zero()
        if predColor then
          outputsColor[1] = torch.Tensor(valBatchSize, 3, 16, 16, 16):cuda():zero()
        end
        outputMasks[1] = torch.Tensor(valBatchSize,1,16,16,16):cuda():zero()
        for l=2,numLevels-1 do
          currSize = outputs[l-1]:size()
          outputs[l] = torch.Tensor(currSize[1],currSize[2],2*currSize[3],2*currSize[4],2*currSize[5]):cuda():zero()
          outputMasks[l] = torch.Tensor(currSize[1],1,2*currSize[3],2*currSize[4],2*currSize[5]):cuda():zero()
          if predColor then
            outputsColor[l] = torch.Tensor(currSize[1],currSize[2],2*currSize[3],2*currSize[4],2*currSize[5]):cuda():zero()
          end
        end
        currSize = outputs[numLevels-1]:size()
        outputs[numLevels] = torch.Tensor(currSize[1],1,2*currSize[3],2*currSize[4],2*currSize[5]):cuda():zero()
        outputMasks[numLevels] = torch.Tensor(currSize[1],1,2*currSize[3],2*currSize[4],2*currSize[5]):cuda():zero()
        if predColor then
          outputsColor[numLevels] = torch.Tensor(currSize[1],currSize[2],2*currSize[3],2*currSize[4],2*currSize[5]):cuda():zero()
        end

        -- run first stage 
        result = net:get(1):forward(input)

        -- run remaining decoder for each element of the batch
        for b=1,valBatchSize do
    
          local function evaluateFull(output, x, y, z, level)
	    if predColor then
	      outputs[level][{{b,b},{},{16*x+1,16*(x+1)},{16*y+1,16*(y+1)},{16*z+1,16*(z+1)}}]:copy(output[{1,1,{},{},{}}])
              outputsColor[level][{{b,b},{},{16*x+1,16*(x+1)},{16*y+1,16*(y+1)},{16*z+1,16*(z+1)}}]:copy(output[{1,{2,4},{},{},{}}])
	    else
	      outputs[level][{{b,b},{},{16*x+1,16*(x+1)},{16*y+1,16*(y+1)},{16*z+1,16*(z+1)}}]:copy(output)
	    end
            outputMasks[level][{{b,b},{},{16*x+1,16*(x+1)},{16*y+1,16*(y+1)},{16*z+1,16*(z+1)}}]:fill(1)
          end
    
          local function evaluateIntermediate(output, feature, x, y, z, level)
        
          local outputSoftMax = volSoftMax:forward(output[{{1,1},{1,3},{},{},{}}])
          
          if predColor then
            outputsColor[level][{{b,b},{},{16*x+1,16*(x+1)},{16*y+1,16*(y+1)},{16*z+1,16*(z+1)}}]:copy(output[{1,{4,6},{},{},{}}])
          end
        
          -- copy output
          outputs[level][{{b,b},{},{16*x+1,16*(x+1)},{16*y+1,16*(y+1)},{16*z+1,16*(z+1)}}]:copy(output[{{1,1},{1,3},{},{},{}}])
          outputMasks[level][{{b,b},{},{16*x+1,16*(x+1)},{16*y+1,16*(y+1)},{16*z+1,16*(z+1)}}]:fill(1)
	  
	  if (level < trainingParams.stopAtLevel) then
        
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
	  
        end
    
          local resultSingleOutput = result[1][{{b,b},{},{},{},{}}]
          local resultSingleFeature = result[2][{{b,b},{},{},{},{}}]
    
          evaluateIntermediate(resultSingleOutput, resultSingleFeature, 0, 0, 0, 1)
        end
        
        
        -- compile the solution
        local softMaxLayer = cudnn.VolumetricSoftMax():cuda()

        outputsCompiled = {}

        local upSampleLayer = nn.VolumetricFullConvolution(1,1,2,2,2,2,2,2,0,0,0)
        upSampleLayer.bias:zero()
        upSampleLayer.weight:fill(1)
        upSampleLayer = upSampleLayer:cuda()

        outputsCompiled = {}
        outputsCompiled[1] = outputs[1]:clone()
        local outputsCompiledColor = {}
        if predColor then
          outputsCompiledColor[1] = outputsColor[1]:clone()
        end
        for l=2,numLevels-1 do
          outputsCompiled[l] = outputs[l]:clone()
    
          local copyMask = torch.eq(outputMasks[l], 0):cuda()
    
          outputsCompiled[l][{{},{1},{},{},{}}]:add(upSampleLayer:forward(outputsCompiled[l-1][{{},{1},{},{},{}}]):cmul(copyMask))
          outputsCompiled[l][{{},{2},{},{},{}}]:add(upSampleLayer:forward(outputsCompiled[l-1][{{},{2},{},{},{}}]):cmul(copyMask))
          outputsCompiled[l][{{},{3},{},{},{}}]:add(upSampleLayer:forward(outputsCompiled[l-1][{{},{3},{},{},{}}]):cmul(copyMask))
          
          if predColor then
            outputsCompiledColor[l] = outputsColor[l]:clone()
            
            outputsCompiledColor[l][{{},{1},{},{},{}}]:add(upSampleLayer:forward(outputsCompiledColor[l-1][{{},{1},{},{},{}}]):cmul(copyMask))
            outputsCompiledColor[l][{{},{2},{},{},{}}]:add(upSampleLayer:forward(outputsCompiledColor[l-1][{{},{2},{},{},{}}]):cmul(copyMask))
            outputsCompiledColor[l][{{},{3},{},{},{}}]:add(upSampleLayer:forward(outputsCompiledColor[l-1][{{},{3},{},{},{}}]):cmul(copyMask))
          end
          
        end
        outputsCompiled[numLevels] = outputs[numLevels]:clone()
        local copyMask = torch.eq(outputMasks[numLevels], 0):cuda()
        local smOutputBefore = softMaxLayer:forward(outputsCompiled[numLevels-1])
        local outputsCompiledTwoLabel = torch.add(smOutputBefore[{{},{2},{},{},{}}], smOutputBefore[{{},{3},{},{},{}}])
        --local outputsCompiledTwoLabel = outputsCompiled[numLevels-1]:clone()
        outputsCompiled[numLevels]:add(upSampleLayer:forward(outputsCompiledTwoLabel):cmul(copyMask))
        
        if predColor then
          outputsCompiledColor[numLevels] = outputsColor[numLevels]:clone()
          
          outputsCompiledColor[numLevels][{{},{1},{},{},{}}]:add(upSampleLayer:forward(outputsCompiledColor[numLevels-1][{{},{1},{},{},{}}]):cmul(copyMask))
          outputsCompiledColor[numLevels][{{},{2},{},{},{}}]:add(upSampleLayer:forward(outputsCompiledColor[numLevels-1][{{},{2},{},{},{}}]):cmul(copyMask))
          outputsCompiledColor[numLevels][{{},{3},{},{},{}}]:add(upSampleLayer:forward(outputsCompiledColor[numLevels-1][{{},{3},{},{},{}}]):cmul(copyMask))
        end
        
        
        -- compile the ground truth
        groundTruths = {}
        local groundTruthColors = {}
        local groundTruthColorMasks = {}
        for l=1,numLevels do
    
          local bISize = blockIndices[l]:size()
          groundTruths[l] = torch.Tensor(1,16*bISize[1],16*bISize[2],16*bISize[3]):cuda()
    
          for bx=1,bISize[1] do
            for by=1,bISize[2] do
              for bz=1,bISize[3] do
                local bI = blockIndices[l][bx][by][bz]
                groundTruths[l][1][{{(bx-1)*16+1,bx*16},{(by-1)*16+1,by*16},{(bz-1)*16+1,bz*16}}]:copy(blocks[l][bI])
              end
            end
          end
          
          if predColor then
            local bISizeColor = voxelColors[l].bi:size()
            groundTruthColors[l] = torch.Tensor(3,16*bISizeColor[1],16*bISizeColor[2],16*bISizeColor[3]):cuda()
            groundTruthColorMasks[l] = torch.Tensor(1,16*bISizeColor[1],16*bISizeColor[2],16*bISizeColor[3]):cuda()
            
            for bx=1,bISizeColor[1] do
              for by=1,bISizeColor[2] do
                for bz=1,bISizeColor[3] do
                  local bI = voxelColors[l].bi[bx][by][bz]
                  groundTruthColors[l][{{},{(bx-1)*16+1,bx*16},{(by-1)*16+1,by*16},{(bz-1)*16+1,bz*16}}]:copy(voxelColors[l].bc[bI])
                  groundTruthColorMasks[l][{{},{(bx-1)*16+1,bx*16},{(by-1)*16+1,by*16},{(bz-1)*16+1,bz*16}}]:copy(voxelColors[l].bm[bI])
                end
              end
            end
          end
        end
        
        -- compute losses
        local colorWeightMultiplier = 1
        for l=1,numLevels-1 do
          valLosses[l] = valLosses[l] + volCECrits[l]:forward(outputsCompiled[l], groundTruths[l])
          
          if predColor then
           
            groundTruthColors[l][1][torch.eq(groundTruthColorMasks[l][1],0)] = outputsCompiledColor[l][1][1][torch.eq(groundTruthColorMasks[l][1],0)]
            groundTruthColors[l][2][torch.eq(groundTruthColorMasks[l][1],0)] = outputsCompiledColor[l][1][2][torch.eq(groundTruthColorMasks[l][1],0)]
            groundTruthColors[l][3][torch.eq(groundTruthColorMasks[l][1],0)] = outputsCompiledColor[l][1][3][torch.eq(groundTruthColorMasks[l][1],0)]
            
            valLossesColor[l] = valLossesColor[l] + mseCrit:forward(outputsCompiledColor[l], groundTruthColors[l])*colorWeightMultiplier
            colorWeightMultiplier = colorWeightMultiplier*2
          end
          
        end
        valLosses[numLevels] = valLosses[numLevels] + bceCrit:forward(outputsCompiled[numLevels], groundTruths[numLevels])
        
        if predColor then
          groundTruthColors[numLevels][1][torch.eq(groundTruthColorMasks[numLevels][1],0)] = outputsCompiledColor[numLevels][1][1][torch.eq(groundTruthColorMasks[numLevels][1],0)]
          groundTruthColors[numLevels][2][torch.eq(groundTruthColorMasks[numLevels][1],0)] = outputsCompiledColor[numLevels][1][2][torch.eq(groundTruthColorMasks[numLevels][1],0)]
          groundTruthColors[numLevels][3][torch.eq(groundTruthColorMasks[numLevels][1],0)] = outputsCompiledColor[numLevels][1][3][torch.eq(groundTruthColorMasks[numLevels][1],0)]
            
          valLossesColor[numLevels] = valLossesColor[numLevels] + mseCrit:forward(outputsCompiledColor[numLevels], groundTruthColors[numLevels])*colorWeightMultiplier
        end
        
        numValidatedElements = numValidatedElements + 1
        
      else
        hierarchical3DShapeNetDataSet:skipNextValExample()
      end
    end
    
    valLosses:div(numValidatedElements)
    if predColor then
      valLossesColor:div(numValidatedElements)
    end
    
    local curEnd = socket.gettime()
    
    if (numLevels == 5) then
      if predColor then
        valLogger:add{i,valLosses[1], valLosses[2], valLosses[3], valLosses[4], valLosses[5],
                              valLossesColor[1], valLossesColor[2], valLossesColor[3], valLossesColor[4], valLossesColor[5], curEnd - start}
      else
        valLogger:add{i,valLosses[1], valLosses[2], valLosses[3], valLosses[4], valLosses[5], curEnd - start}
      end
    elseif (numLevels == 2) then
      valLogger:add{i,valLosses[1], valLosses[2], curEnd - start}
    end
    
    
    net:training()
    
  end
end
