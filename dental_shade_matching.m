 function DeltaE()
try
    fontSize = 14;
    image=imread('D:\College\Dental Shade Matching 2\Patients Photos\00001.jpg');
    % figure,imshow(image);    
    
    gray_image=rgb2gray(image);
   
    lmean=conv2(gray_image,ones(3)/9,'same');
    lstd=stdfilt(gray_image);
    % data(:,1)=image(:);
    data(:,1)=lmean(:);
    data(:,2)=lstd(:);
    D(:,1)=abs(data(:,1)-data(:,2));
    D(:,2)=abs(data(:,1)+data(:,2));
    data=double(D);
%Intialize the cluster number%
cluster=3;

%Call FCM algorithm to cluster the input image%
%[center,MF,obj]=fcm(data,cluster);
[MF,Centers,obj,res]=FCM_EU(data,cluster);

%Calculate Cluster validity index%
RF1=0;
Vpec=0;
for j=1:size(MF,2)
    for i=1:cluster
            RF1 =RF1+ (MF(i,j)^2);
            Vpec = Vpec + MF(i,j)*log(MF(i,j));
    end
end
RF1= RF1/size(MF,2);
Vpc1(cluster)= RF1;
Vpe1 = (-Vpec)/size(MF,2);
Vpe(cluster)= Vpe1;
Vfs(cluster)= res.Vfs ;
Vxb(cluster) = res.vxb ;


%Display the image%
figure
subplot(231); imshow(image,[])  
for i=1:cluster
    imgfi=reshape((MF(i,:,:)),size(image,1),size(image,2));
    subplot(2,3,i+1); imshow(imgfi,[])
    title(['Index No: ' int2str(i)])
end

prompt = {'Enter your option:'};
dlgtitle = 'Input';
dims = [1 35];
definput = {'0'};
answer = inputdlg(prompt,dlgtitle,dims,definput)
i = str2double(answer);
imfgi=reshape((MF(i,:,:)),size(image,1),size(image,2));
imfgi=medfilt2(imfgi,[5 5]);

newi=im2bw(imfgi,graythresh(imfgi));
se=strel('square',15);
iopenned=imopen(newi,se);
iopen=bwareaopen(iopenned,10000);
figure,imshow(iopen);
    
maskedRgbImage = bsxfun(@times, image, cast(iopen, 'like', image));
figure,imshow(maskedRgbImage);

j=imcrop(maskedRgbImage);
figure,imshow(j);


%removing flashlight reflection
% grayj=rgb2gray(j);
%     mask = grayj > 300; % whatever value works.
%     fixedImage = regionfill(grayj, mask);
%     figure,imshow(fixedImage);

% [rows columns numberOfColorBands] = size(j);
% % divide an image up into blocks by using mat2cell().
% blockSizeR = 150; % Rows in block.
% blockSizeC = 100; % Columns in block.
% % Figure out the size of each block in rows. 
% % Most will be blockSizeR but there may be a remainder amount of less than that.
% wholeBlockRows = floor(rows / blockSizeR);
% blockVectorR = [blockSizeR * ones(1, wholeBlockRows), rem(rows, blockSizeR)];
% % Figure out the size of each block in columns. 
% wholeBlockCols = floor(columns / blockSizeC);
% blockVectorC = [blockSizeC * ones(1, wholeBlockCols), rem(columns, blockSizeC)];
% % Create the cell array, ca.  
% % Each cell (except for the remainder cells at the end of the image)
% % in the array contains a blockSizeR by blockSizeC by 3 color array.
% % This line is where the image is actually divided up into blocks.
% if numberOfColorBands > 1
%   % It's a color image.
%   ca = mat2cell(j, blockVectorR, blockVectorC, numberOfColorBands);
% else
%   ca = mat2cell(rgbImage, blockVectorR, blockVectorC);
% end
% % Now display all the blocks.
% plotIndex = 1;
% numPlotsR = size(ca, 1);
% numPlotsC = size(ca, 2);
% for r = 1 : numPlotsR
%   for c = 1 : numPlotsC
%     fprintf('plotindex = %d,   c=%d, r=%d\n', plotIndex, c, r);
%     % Specify the location for display of the image.
%     subplot(numPlotsR, numPlotsC, plotIndex);
%     % Extract the numerical array out of the cell
%     % just for tutorial purposes.
%     rgbBlock = ca{r,c};
%     imshow(rgbBlock); % Could call imshow(ca{r,c}) if you wanted to.
%     [rowsB columnsB numberOfColorBandsB] = size(rgbBlock);
%     % Make the caption the block number.
%     caption = sprintf('Block #%d of %d\n%d rows by %d columns', ...
%       plotIndex, numPlotsR*numPlotsC, rowsB, columnsB);
%     title(caption);
%     drawnow;
%     % Increment the subplot to the next location.
%     plotIndex = plotIndex + 1;
%   end
% end
% % Display the original image in the upper left.
% % subplot(4, 6, 1);
% 
% % title('Original Image');

% Convert image from RGB colorspace to lab color space.
	cform = makecform('srgb2lab');
	lab_Image = applycform(im2double(j),cform);
	
	% Extract out the color bands from the original image
	% into 3 separate 2D arrays, one for each color component.
	LChannel = lab_Image(:, :, 1); 
	aChannel = lab_Image(:, :, 2); 
	bChannel = lab_Image(:, :, 3); 
	
	% Display the lab images.
	subplot(3, 4, 2);
	imshow(LChannel, []);
	title('L Channel', 'FontSize', fontSize);
	subplot(3, 4, 3);
	imshow(aChannel, []);
	title('a Channel', 'FontSize', fontSize);
	subplot(3, 4, 4);
	imshow(bChannel, []);
	title('b Channel', 'FontSize', fontSize);


% Get the average lab color value.
	[LMean, aMean, bMean] = GetMeanLABValues(LChannel, aChannel, bChannel, j);
% Make uniform images of only that one single LAB color.
	LStandard = LMean * ones(rows, columns);
	aStandard = aMean * ones(rows, columns);
	bStandard = bMean * ones(rows, columns);
	
	% Create the delta images: delta L, delta A, and delta B.
	deltaL = LChannel - LStandard;
	deltaa = aChannel - aStandard;
	deltab = bChannel - bStandard;

% Create the Delta E image.
	% This is an image that represents the color difference.
	% Delta E is the square root of the sum of the squares of the delta images.
	deltaE = sqrt(deltaL .^ 2 + deltaa .^ 2 + deltab .^ 2);
	
	% Mask it to get the Delta E in the mask region only.
	maskedDeltaE = deltaE .* j;
	% Get the mean delta E in the mask region
	% Note: deltaE(mask) is a 1D vector of ONLY the pixel values within the masked area.
 	meanMaskedDeltaE = mean(deltaE(j));
	% Get the standard deviation of the delta E in the mask region
	stDevMaskedDeltaE = std(deltaE(j));
	message = sprintf('The mean LAB = (%.2f, %.2f, %.2f).\nThe mean Delta E in the masked region is %.2f +/- %.2f',...
		LMean, aMean, bMean, meanMaskedDeltaE, stDevMaskedDeltaE);	
	
	% Display the masked Delta E image - the delta E within the masked region only.
	subplot(3, 4, 6);
	imshow(maskedDeltaE, []);
	caption = sprintf('Delta E between image within masked region\nand mean color within masked region.\n(With amplified intensity)');
	title(caption, 'FontSize', fontSize);

	% Display the Delta E image - the delta E over the entire image.
	subplot(3, 4, 7);
	imshow(deltaE, []);
	caption = sprintf('Delta E Image\n(Darker = Better Match)');
	title(caption, 'FontSize', fontSize);


catch ME
	errorMessage = sprintf('Error running this m-file:\n%s\n\nThe error message is:\n%s', ...
		mfilename('fullpath'), ME.message);
	errordlg(errorMessage);
end
return;



%-----------------------------------------------------------------------------
% Get the average lab within the mask region.
function [LMean, aMean, bMean] = GetMeanLABValues(LChannel, aChannel, bChannel, mask)
try
	LVector = LChannel(mask); % 1D vector of only the pixels within the masked area.
	LMean = mean(LVector);
	aVector = aChannel(mask); % 1D vector of only the pixels within the masked area.
	aMean = mean(aVector);
	bVector = bChannel(mask); % 1D vector of only the pixels within the masked area.
	bMean = mean(bVector);
catch ME
	errorMessage = sprintf('Error running GetMeanLABValues:\n\n\nThe error message is:\n%s', ...
		ME.message);
	WarnUser(errorMessage);
end
return; % from GetMeanLABValues

