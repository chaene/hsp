local matio = require('matio')

math.randomseed( os.time() )

hierarchical3DShapeNetDataSet = {}
hierarchical3DShapeNetDataSet.resolutionStr = "256"
hierarchical3DShapeNetDataSet.numViews = 20

hierarchical3DShapeNetDataSet.testDataFiles = {}
hierarchical3DShapeNetDataSet.trainDataFiles = {}
hierarchical3DShapeNetDataSet.valDataFiles = {}

hierarchical3DShapeNetDataSet.lastTrainInd = 0
hierarchical3DShapeNetDataSet.lastValInd = 0
hierarchical3DShapeNetDataSet.lastTestInd = 0

hierarchical3DShapeNetDataSet.useColor = false

hierarchical3DShapeNetDataSet.loadVoxelColors = false

function hierarchical3DShapeNetDataSet:setPath(path)
  self.path = path
end

function hierarchical3DShapeNetDataSet:setResolution(resolutionStr)
  self.resolutionStr = resolutionStr
end

function hierarchical3DShapeNetDataSet:setColor()
  self.useColor = true
end

function hierarchical3DShapeNetDataSet:setLoadVoxelColors()
  self.loadVoxelColors = true
end


function hierarchical3DShapeNetDataSet:addTestDataFromClass(class)
    io.input(self.path .. "/test_" .. class .. ".txt")
    for line in io.lines() do
      table.insert(self.testDataFiles, class .. "/" .. line)
    end
end

function hierarchical3DShapeNetDataSet:addValDataFromClass(class)
    io.input(self.path .. "/val_" .. class .. ".txt")
    for line in io.lines() do
        table.insert(self.valDataFiles, class .. "/" .. line)
    end
end

function hierarchical3DShapeNetDataSet:addTrainDataFromClass(class)
    io.input(self.path .. "/train_" .. class .. ".txt")
    for line in io.lines() do
        table.insert(self.trainDataFiles, class .. "/" .. line)
    end
end

function hierarchical3DShapeNetDataSet:shuffleTrainDataFileNames()
  
  rP = torch.randperm(#self.trainDataFiles)
  
  trainDataFilesPerm = {}
  for i = 1,#self.trainDataFiles do
    trainDataFilesPerm[i] = self.trainDataFiles[rP[i]]
  end
  
  self.trainDataFiles = trainDataFilesPerm
end

function hierarchical3DShapeNetDataSet:loadSample(fileNames, modelID, viewID)
  
  local obs 
  local blockIndices = {}
  local blocks = {}
  local modelName = string.sub(fileNames[modelID],1,-5)
  local voxelColors = {}
    
  if self.useColor then
    local fileName = self.path .. "/blenderRenderPreprocess/" .. string.sub(fileNames[modelID] ,1, -5) .. "/" .. "render_" .. viewID-1 .. ".png"
	
    -- white background
    local img = image.load(fileName)
    img = image.scale(img, 128, 128)
    local alphaMask = img[4]:repeatTensor(3,1,1)
    img = torch.cmul(img:narrow(1,1,3),alphaMask) + 1 - alphaMask
    obs = img
  else
    local fileName = self.path .. "/blenderRenderPreprocess/"  .. string.sub(fileNames[modelID] ,1, -5) .. "/" .. "depth_" .. viewID-1 .. ".png"
    local dm = image.load(fileName)
    dm = image.scale(dm, 128, 128, 'simple')*10 - 2
    dm[torch.gt(dm, 0.866)] = 0.866
    obs = dm
  end
  
  if self.resolutionStr == "256" then
    local pyramid = matio.load(self.path .. "/modelBlockedPyramid" .. self.resolutionStr 
                              .."/" .. string.sub(fileNames[modelID],1,-5) .. ".mat", {'b1','b2','b3','b4', 'bi1', 'bi2', 'bi3', 'bi4'})
                              
    local voxels = matio.load(self.path .. "/modelBlockedVoxels".. self.resolutionStr
                                .."/" .. string.sub(fileNames[modelID],1,-5) .. ".mat", {'b','bi'})
                                
    blockIndices[1] = pyramid.bi1
    blockIndices[2] = pyramid.bi2
    blockIndices[3] = pyramid.bi3
    blockIndices[4] = pyramid.bi4
    blockIndices[5] = voxels.bi
    
    blocks[1] = pyramid.b1
    blocks[2] = pyramid.b2
    blocks[3] = pyramid.b3
    blocks[4] = pyramid.b4
    blocks[5] = voxels.b
  elseif self.resolutionStr == "32" then
        local pyramid = matio.load(self.path .. "/modelBlockedPyramid" .. self.resolutionStr 
                              .."/" .. string.sub(fileNames[modelID],1,-5) .. ".mat", {'b1', 'bi1'})
                              
    local voxels = matio.load(self.path .. "/modelBlockedVoxels".. self.resolutionStr
                                .."/" .. string.sub(fileNames[modelID],1,-5) .. ".mat", {'b','bi'})
                                
    blockIndices[1] = pyramid.bi1
    blockIndices[2] = voxels.bi
    
    blocks[1] = pyramid.b1
    blocks[2] = voxels.b
  end
  
  if self.loadVoxelColors then
    if self.resolutionStr == "256" then
      pyramidColors = matio.load(self.path .. "/modelBlockedPyramidColors" .. self.resolutionStr 
                                .."/" .. string.sub(fileNames[modelID],1,-5) .. ".mat", {'bi1', 'bm1', 'bc1', 
                                              'bi2', 'bm2', 'bc2', 'bi3', 'bm3', 'bc3', 'bi4', 'bm4', 'bc4', 'bi5', 'bm5', 'bc5'})
      
      voxelColors[1] = {}
      voxelColors[1].bi = pyramidColors.bi1
      voxelColors[1].bm = pyramidColors.bm1
      voxelColors[1].bc = pyramidColors.bc1  
      voxelColors[2] = {}
      voxelColors[2].bi = pyramidColors.bi2
      voxelColors[2].bm = pyramidColors.bm2
      voxelColors[2].bc = pyramidColors.bc2  
      voxelColors[3] = {}
      voxelColors[3].bi = pyramidColors.bi3
      voxelColors[3].bm = pyramidColors.bm3
      voxelColors[3].bc = pyramidColors.bc3 
      voxelColors[4] = {}
      voxelColors[4].bi = pyramidColors.bi4
      voxelColors[4].bm = pyramidColors.bm4
      voxelColors[4].bc = pyramidColors.bc4
      voxelColors[5] = {}
      voxelColors[5].bi = pyramidColors.bi5
      voxelColors[5].bm = pyramidColors.bm5
      voxelColors[5].bc = pyramidColors.bc5
    end
  end
  
  return obs, blockIndices, blocks, modelName, voxelColors
end

function hierarchical3DShapeNetDataSet:getNextSample(fileNames, last, randomizeElement, randomizeView)
  
  local ind = 0
  if (randomizeElement) then
    ind = math.random(fileNames)
  else
    last = last + 1
    if last > #fileNames then
      last = 1
    end
    ind = last
  end
  
  local view = 1
  if (randomizeView) then
    view = math.random(hierarchical3DShapeNetDataSet.numViews)
  end
  
  local obs, blockIndices, blocks, modelName, voxelColors = self:loadSample(fileNames, ind, view)
  
  return ind, obs, blockIndices, blocks, modelName, voxelColors
end

function hierarchical3DShapeNetDataSet:getNextTrainExample(randomizeElement, randomizeView)
  
  local ind, obs, blockIndices, blocks, modelName, voxelColors = self:getNextSample(self.trainDataFiles, self.lastTrainInd, randomizeElement, randomizeView)
  --if (ind > 100) then
  --  ind = 0
  --end
  self.lastTrainInd = ind

  
  return obs, blockIndices, blocks, modelName, voxelColors
end


function hierarchical3DShapeNetDataSet:getNextValExample(randomize, randomizeView)
  
  local ind, obs, blockIndices, blocks, modelName, voxelColors = self:getNextSample(self.valDataFiles, self.lastValInd, randomizeElement, randomizeView)
  self.lastValInd = ind
  
  return obs, blockIndices, blocks, modelName, voxelColors
end

-- useful to validate on subset
function hierarchical3DShapeNetDataSet:skipNextValExample()
  self.lastValInd = self.lastValInd + 1
    if self.lastValInd > #self.valDataFiles then
      self.lastValInd = 1
    end
    ind = self.lastValInd
end

function hierarchical3DShapeNetDataSet:getNextTestExample(randomize, randomizeView)
  
  local ind, obs, blockIndices, blocks, modelName, voxelColors = self:getNextSample(self.testDataFiles, self.lastTestInd, randomizeElement, randomizeView)
  self.lastTestInd = ind
  
  return obs, blockIndices, blocks, modelName, voxelColors
end









