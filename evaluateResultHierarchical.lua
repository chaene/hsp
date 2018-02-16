require 'image'
require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'optim'
require 'cudnn'
dofile('utils/instanceNormalization.lua')
require 'lfs'

local hsp = require 'hsp'
local matio = require 'matio'

function split(inputstr, sep)
  if sep == nil then
    sep = "%s"
  end
  local t={} ; i=1
  for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
    t[i] = str
    i = i + 1
  end
  return t
end

networkParameters, evalParams  = dofile(arg[2])
cutorch.setDevice(arg[1])

if (networkParameters.padding == nil) then
  error("networkParameters.padding needs to be set to true or false")
end

if (evalParams.resolutionStr == nil) then
  error("evalParams.resolutionStr as to be set")
end

local predColor = networkParameters.predictColors

local splitStr = evalParams.split
local objStr = ""
if evalParams.saveObj then
  objStr = "_obj"
end

local metric = evalParams.metric

local padding = networkParameters.padding

dofile('utils/hierarchicalShapenetDataLoader.lua')
hierarchical3DShapeNetDataSet:setResolution(evalParams.resolutionStr)

if networkParameters.useColor then
  hierarchical3DShapeNetDataSet:setColor()
end

if predColor then
  hierarchical3DShapeNetDataSet:setLoadVoxelColors()
end

hierarchical3DShapeNetDataSet:setPath(evalParams.datasetFolder)
for c=1,#evalParams.classes do
  if splitStr == "val" then
    hierarchical3DShapeNetDataSet:addValDataFromClass(evalParams.classes[c])
  elseif splitStr == "test" then
    hierarchical3DShapeNetDataSet:addTestDataFromClass(evalParams.classes[c])
  end
end

local peakPerformance = 0
local peakPerformanceIter = 0
local peakPerformanceThresh = 0
if metric == "CD" then
  peakPerformance = 10 -- value which is bigger than biggest CD
end


local splitThreshold = networkParameters.splitThreshold

local volSoftMax = cudnn.VolumetricSoftMax():cuda()

local thresholds = evalParams.thresholds

local numLevels = networkParameters.numLevels

if (numLevels < 2) then
  error("numLevels needs to be >= 2")
end

local gridRes = 8*math.pow(2,numLevels)


local AvGIoUhighResAvgFile = assert(io.open(evalParams.outputFolder .. "/evaluation_" .. splitStr .. objStr .."_AvG" .. metric .. "_Avg_highRes_" .. evalParams.snapshotStartIter .. ".txt", "w"))
AvGIoUhighResAvgFile:write("iter")
for t=1,#thresholds do
  AvGIoUhighResAvgFile:write(", " .. thresholds[t])
end
AvGIoUhighResAvgFile:write("\n")

local AvGIoUhighResFiles = {}
for c=1,#evalParams.classes do
  AvGIoUhighResFiles[evalParams.classes[c]] = assert(io.open(evalParams.outputFolder .. "/evaluation_" .. splitStr .. objStr .. "_AvG" .. metric .. "_" .. evalParams.classes[c] .. "_highRes_" .. evalParams.snapshotStartIter .. ".txt", "w"))
  AvGIoUhighResFiles[evalParams.classes[c]]:write("iter")
  for t=1,#thresholds do
    AvGIoUhighResFiles[evalParams.classes[c]]:write(", " .. thresholds[t])
  end
  AvGIoUhighResFiles[evalParams.classes[c]]:write("\n")
end

--generate obj folders if objs output is written
 local meshFolderBaseName
if (evalParams.saveObj == true) then
  if predColor then
    meshFolderBaseName = "col_plys"
  else
    meshFolderBaseName = "objs"
  end
  
  local objFolderExists = lfs.attributes(evalParams.outputFolder .. "/" .. meshFolderBaseName .. "_" .. splitStr .. "/",'modification')
  if (objFolderExists == nil) then
    lfs.mkdir(evalParams.outputFolder .. "/" ..meshFolderBaseName .. "_" .. splitStr .. "/",'modification')
  end
  for c=1,#evalParams.classes do
    local classFolderExists = lfs.attributes(evalParams.outputFolder .. "/" .. meshFolderBaseName .. "_" .. splitStr .. "/" .. evalParams.classes[c], 'modification')
    if (classFolderExists == nil) then
      lfs.mkdir(evalParams.outputFolder .. "/" .. meshFolderBaseName .. "_" .. splitStr .. "/" .. evalParams.classes[c])
    end
  end
end

local numPredVoxels = {}
if (evalParams.computeNumPredVoxels) then
  for c=1,#evalParams.classes do
    numPredVoxels[evalParams.classes[c]] = torch.Tensor(numLevels):zero()
  end
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

for iter=evalParams.snapshotStartIter,evalParams.snapshotEndIter,evalParams.snapshotInterval do
  
    local totalIoUHR = {}
    local numEvaluatedElements = {}
    for c=1,#evalParams.classes do
      totalIoUHR[evalParams.classes[c]] = {}
      numEvaluatedElements[evalParams.classes[c]] = 0
      for t=1,#thresholds do
        totalIoUHR[evalParams.classes[c]][t] = 0
      end
    end
  
  if not evalParams.loadCache then
    net = torch.load(evalParams.highResSnapshotPath .. "/net" .. iter .. ".t7")
    net:evaluate()
  
    -- open output file
    local IoUhighResFile = assert(io.open(evalParams.outputFolder .. "/evaluation_" .. splitStr .. objStr .. "_" .. metric .. "_highRes_" .. iter .. ".txt", "w"))
    IoUhighResFile:write("modelName")
    for t=1,#thresholds do
      IoUhighResFile:write(", " .. thresholds[t])
    end
    IoUhighResFile:write("\n")
    
    local numEvalFiles = 0
    if splitStr == "val" then
      numEvalFiles = #hierarchical3DShapeNetDataSet.valDataFiles
    else
      numEvalFiles = #hierarchical3DShapeNetDataSet.testDataFiles
    end
  

    for j=1,numEvalFiles do
      if ((j-1)%evalParams.subsampling == 0) then
        local obs, blockIndices, blocks, modelName
	if splitStr == "val" then
	  obs, blockIndices, blocks, modelName = hierarchical3DShapeNetDataSet:getNextValExample(false, false)
	else
	  obs, blockIndices, blocks, modelName = hierarchical3DShapeNetDataSet:getNextTestExample(false, false)
	end
	
	class = split(modelName, "/")[1]
        
        IoUhighResFile:write(modelName .. " ")
        
	if networkParameters.useColor then
	  input = obs:view(1,3,128,128):cuda()
	else
	  input = obs:view(1,1,128,128):cuda()
	end
	  
        for l=1,numLevels do
          blocks[l] = blocks[l]:cuda()
        end
	  
        print("Iteration " .. iter .. " running model " .. j .. " : " .. modelName)
	  
        -- forward pass through the network
        outputs = {}
        outputMasks = {}
        outputsColor = {}
        outputs[1] = torch.Tensor(1,3,16,16,16):cuda():zero()
        outputMasks[1] = torch.Tensor(1,1,16,16,16):cuda():zero()
        if predColor then
          outputsColor[1] = torch.Tensor(1, 3, 16, 16, 16):cuda():zero()
        end
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
	curNumPredVoxels = torch.Tensor(numLevels):zero()

        -- run first stage 
        result = net:get(1):forward(input)

        local function evaluateFull(output, x, y, z, level)
          
          if predColor then
            outputs[level][{{1,1},{},{16*x+1,16*(x+1)},{16*y+1,16*(y+1)},{16*z+1,16*(z+1)}}]:copy(output[{1,1,{},{},{}}])
            outputsColor[level][{{1,1},{},{16*x+1,16*(x+1)},{16*y+1,16*(y+1)},{16*z+1,16*(z+1)}}]:copy(output[{1,{2,4},{},{},{}}])
          else
            outputs[level][{{1,1},{},{16*x+1,16*(x+1)},{16*y+1,16*(y+1)},{16*z+1,16*(z+1)}}]:copy(output)
          end
          
	  if evalParams.computeNumPredVoxels then
	    curNumPredVoxels[level] = curNumPredVoxels[level] + 16*16*16
	  end
          
          outputMasks[level][{{1,1},{},{16*x+1,16*(x+1)},{16*y+1,16*(y+1)},{16*z+1,16*(z+1)}}]:fill(1)
        end
      
        local function evaluateIntermediate(output, feature, x, y, z, level)
          
          if predColor then
            outputsColor[level][{{1,1},{},{16*x+1,16*(x+1)},{16*y+1,16*(y+1)},{16*z+1,16*(z+1)}}]:copy(output[{1,{4,6},{},{},{}}])
          end
	  
	  if evalParams.computeNumPredVoxels then
	    curNumPredVoxels[level] = curNumPredVoxels[level] + 16*16*16
	  end
	  
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
      
        numEvaluatedElements[class] = numEvaluatedElements[class] + 1
	  
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
        outputsCompiled[numLevels]:add(upSampleLayer:forward(outputsCompiledTwoLabel):cmul(copyMask))
        
        if predColor then
          outputsCompiledColor[numLevels] = outputsColor[numLevels]:clone()
          
          outputsCompiledColor[numLevels][{{},{1},{},{},{}}]:add(upSampleLayer:forward(outputsCompiledColor[numLevels-1][{{},{1},{},{},{}}]):cmul(copyMask))
          outputsCompiledColor[numLevels][{{},{2},{},{},{}}]:add(upSampleLayer:forward(outputsCompiledColor[numLevels-1][{{},{2},{},{},{}}]):cmul(copyMask))
          outputsCompiledColor[numLevels][{{},{3},{},{},{}}]:add(upSampleLayer:forward(outputsCompiledColor[numLevels-1][{{},{3},{},{},{}}]):cmul(copyMask))
        end
	  
	  
        -- compile the ground truth
        groundTruths = {}
        for l=1,numLevels do
          local bISize = blockIndices[l]:size()
          groundTruths[l] = torch.Tensor(1,16*bISize[1],16*bISize[2],16*bISize[3]):cuda()

          for bx=1,bISize[1] do
            for by=1,bISize[2] do
              for bz=1,bISize[3] do
                local bI = blockIndices[l][bx][by][bz]
                groundTruths[l][1][{{(bx-1)*16+1,bx*16},{(by-1)*16+1,by*16},{(bz-1)*16+1,bz*16}}]:copy(blocks[l][bI],1,2)
              end
            end
          end
        end
	  
        -- compute Error Measures-
        local labelBoundary
        local labelEDT
        
        if metric == "CD" then
          labelBoundary = groundTruths[numLevels]:clone():view(gridRes,gridRes,gridRes):double()
          hsp.boundary(labelBoundary)
          
          labelEDT = groundTruths[numLevels]:clone():view(gridRes, gridRes, gridRes):double()
          hsp.boundaryEDT(labelEDT)
        end

        for t=1,#thresholds do 
          local threshold = thresholds[t]
        
          local binarizedResult = outputsCompiled[numLevels]:clone()
          binarizedResult[torch.le(binarizedResult, threshold)] = 0
          binarizedResult[torch.ge(binarizedResult, threshold)] = 1
	  
	  if (metric == "CD") then
	    if torch.max(binarizedResult) < 1 then
	      local maxVal = torch.max(outputsCompiled[numLevels])
	      binarizedResult[torch.ge(outputsCompiled[numLevels],maxVal-1e-5)] = 1
	    end
	  end
        
          -- compute IoU
          local measure
          if metric == "IoU" then
            local unionHR = torch.cmax(binarizedResult, groundTruths[numLevels])
            local intersectionHR = torch.cmin(binarizedResult, groundTruths[numLevels])
            local IoUHR = torch.sum(intersectionHR)/torch.sum(unionHR)
            measure = IoUHR
          elseif metric == "CD" then
            local outputVolBoundary = binarizedResult:clone():view(gridRes, gridRes, gridRes):double()
            hsp.boundary(outputVolBoundary)

            local outputEDT = binarizedResult:clone():view(gridRes, gridRes, gridRes):double()
            hsp.boundaryEDT(outputEDT)
            
            local numOutputVolBoundary = torch.sum(outputVolBoundary)
            local numLabelBoundary = torch.sum(labelBoundary)

            --print(torch.max(labelEDT))
            
            
            if (numOutputVolBoundary > 0 and numLabelBoundary > 0) then
                local cd1 = torch.sum(torch.cmul(outputVolBoundary, labelEDT))/(gridRes*numOutputVolBoundary)
                local cd2 = torch.sum(torch.cmul(labelBoundary, outputEDT))/(gridRes*numLabelBoundary)
                --print("cd1 = " .. cd1 .. "  cd2 = " .. cd2)
                measure = (cd1 + cd2)/2
            else
              measure = math.sqrt(3)/2.0
            end
            --print(measure)
          end
        
          IoUhighResFile:write(measure .. " ")
        
          totalIoUHR[class][t] = totalIoUHR[class][t] + measure
        end
      
	if evalParams.computeNumPredVoxels then
	    numPredVoxels[class]:add(curNumPredVoxels)
	end
      
        IoUhighResFile:write("\n")
        IoUhighResFile:flush()
           
      
        if (evalParams.saveObj == true) then
          
          local folderString = "objs_"
        
	  if predColor then
            for ol=1,5 do
              if ol < 5 then
                local currOutput = outputsCompiled[ol][1]:clone()
                currOutput = softMaxLayer:forward(currOutput)
                currOutput[2]:add(currOutput[3])
                hsp.saveColoredMeshAsPLY(currOutput[2]:double(),outputsCompiledColor[ol]:double()[{1,{},{},{},{}}],evalParams.marchingCubesThreshold, evalParams.outputFolder .. "/col_plys_" .. splitStr .."/" .. modelName .. "_" .. ol .. ".ply") 
              else
                hsp.saveColoredMeshAsPLY(outputsCompiled[ol][{1,1,{},{},{}}]:double(),outputsCompiledColor[ol]:double()[{1,{},{},{},{}}],evalParams.marchingCubesThreshold, evalParams.outputFolder ..  "/col_plys_" .. splitStr .."/" .. modelName .. ".ply") 
              end
            end
            folderString = "col_plys_"
	  else
	    hsp.saveMeshAsObj(outputsCompiled[numLevels][{1,1,{},{},{}}]:double(),evalParams.marchingCubesThreshold, evalParams.outputFolder .. "/objs_" .. splitStr .."/" .. modelName .. ".obj")
	  end
	  
          if networkParameters.useColor then
            image.save(evalParams.outputFolder .. "/" .. folderString .. splitStr .."/" .. modelName .. "_input.png", obs)
          else
            obs = obs + 0.866
            obs = obs:div(2*0.866)
            image.save(evalParams.outputFolder .. "/" .. folderString .. splitStr .."/" .. modelName .. "_input.png", obs)
          end
        end
      else
	if splitStr == "val" then
	  hierarchical3DShapeNetDataSet:skipNextValExample()
	else
	  hierarchical3DShapeNetDataSet:skipNextTestExample()
	end
      end
    end
  else  
    
    -- loading the data from the valDataFiles, computing statistics
    local IoUhighResFile = assert(io.open(evalParams.outputFolder .. "/evaluation_" .. splitStr .. "_" .. metric .. "_highRes_" .. iter .. ".txt", "r"))
    titleLine = true
    
    for line in IoUhighResFile:lines() do
      if titleLine then
        titleLine = false
      else
        -- reading data
        elements = split(line, " ")
        class = split(elements[1], "/")[1]
        print (elements[1])
        for t=1,#thresholds do
          totalIoUHR[class][t] = totalIoUHR[class][t] + tonumber(elements[t+1])
        end
        numEvaluatedElements[class] = numEvaluatedElements[class] + 1
      end
    end
  end
  
  local IoUAvgs = {}
  for t=1,#thresholds do
    IoUAvgs[t] = 0
  end
  
  for c=1,#evalParams.classes do
  
    AvGIoUhighResFiles[evalParams.classes[c]]:write(iter) 
    
    for t=1,#thresholds do
      local classIoU = totalIoUHR[evalParams.classes[c]][t] / numEvaluatedElements[evalParams.classes[c]]
      AvGIoUhighResFiles[evalParams.classes[c]]:write(" " .. classIoU)
      IoUAvgs[t] = IoUAvgs[t] + classIoU
    end
    AvGIoUhighResFiles[evalParams.classes[c]]:write("\n")
    AvGIoUhighResFiles[evalParams.classes[c]]:flush()
  end
  
  AvGIoUhighResAvgFile:write(iter)
  for t=1,#thresholds do
    local IoUAvg = IoUAvgs[t]/#evalParams.classes
    AvGIoUhighResAvgFile:write(" " .. IoUAvg)
    if metric == "IoU" then
      if (IoUAvg > peakPerformance) then
        peakPerformance = IoUAvg
        peakPerformanceIter = iter
        peakPerformanceThresh = thresholds[t]
      end
    elseif metric == "CD" then
      if (IoUAvg < peakPerformance) then
        peakPerformance = IoUAvg
        peakPerformanceIter = iter
        peakPerformanceThresh = thresholds[t]
      end
    end
  end
  AvGIoUhighResAvgFile:write("\n")
  AvGIoUhighResAvgFile:flush()
  
  if evalParams.computeNumPredVoxels then
    for c=1,#evalParams.classes do
      local classString = evalParams.classes[c]
      for l=1,numLevels do
        classString = classString .. " " .. numPredVoxels[evalParams.classes[c]][l]/numEvaluatedElements[evalParams.classes[c]]
      end
      print(classString)
    end
  end
  
end
-- 
print("peakPerformance = " .. peakPerformance)
print("peakPerformanceIter = " .. peakPerformanceIter)
print("peakPerformanceThresh = " .. peakPerformanceThresh)