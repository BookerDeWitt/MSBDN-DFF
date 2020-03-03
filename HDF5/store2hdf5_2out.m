function [curr_dat_sz,  curr_lab_sz] = store2hdf5_2out(filename, data, labels, create, startloc, chunksz)  
  % *data* is W*H*C*N matrix of images should be normalized (e.g. to lie between 0 and 1) beforehand
  % *label* is D*N matrix of labels (D labels per sample) 
  % *create* [0/1] specifies whether to create file newly or to append to previously created file, useful to store information in batches when a dataset is too big to be held in memory  (default: 1)
  % *startloc* (point at which to start writing data). By default, 
  % if create=1 (create mode), startloc.data=[1 1 1 1], and startloc.lab=[1 1]; 
  % if create=0 (append mode), startloc.data=[1 1 1 K+1], and startloc.lab = [1 K+1]; where K is the current number of samples stored in the HDF
  % chunksz (used only in create mode), specifies number of samples to be stored per chunk (see HDF5 documentation on chunking) for creating HDF5 files with unbounded maximum size - TLDR; higher chunk sizes allow faster read-write operations 

  % verify that format is right
  dat_dims=size(data);          %%data* is W*H*C*N
  lab_dims=size(labels);
  num_samples=dat_dims(end); %dat_dims最后一维，即N

  assert(lab_dims(end)==num_samples, 'Number of samples should be matched between data and labels');%数据和标签数应该相等
 

  
  if ~exist('create','var')  %不存在时需创建
    create=true;
  end

  
  if create      %创建模式
    %fprintf('Creating dataset with %d samples\n', num_samples);
    if ~exist('chunksz', 'var')
      chunksz=1000;
    end
    if exist(filename, 'file')
      fprintf('Warning: replacing existing file %s \n', filename);
      delete(filename);
    end     
    
    %创建压缩数据集，filename为HDF5文件名，/data为要创建HDF5文件的数据集名称，
    %[dat_dims(1:end-1)Inf]数据集大小，'Datatype', 'single'数据类型单精度
    %'ChunkSize', [dat_dims(1:end-1) chunksz])分块大小
    h5create(filename, '/data', [dat_dims(1:end-1) Inf], 'Datatype', 'single', 'ChunkSize', [dat_dims(1:end-1) chunksz]); % width, height, channels, number
    
    h5create(filename, '/label', [lab_dims(1:end-1) Inf], 'Datatype', 'single', 'ChunkSize', [lab_dims(1:end-1) chunksz]); % width, height, channels, number 
    if ~exist('startloc','var')   %不存在时初始化为全1矩阵
      startloc.dat=[ones(1,length(dat_dims)-1), 1];  %创建length(dat_dims)-1维全1矩阵，然后在最后添加1
     
      startloc.lab=[ones(1,length(lab_dims)-1), 1];
    end 
    
  else  % append mode追加模式
    if ~exist('startloc','var')
      info=h5info(filename);   %返回有关 filename 指定的整个 HDF5 文件的信息
      prev_dat_sz=info.Datasets(1).Dataspace.Size;
     
      prev_lab_sz=info.Datasets(2).Dataspace.Size;
      
      assert(prev_dat_sz(1:end-1)==dat_dims(1:end-1), 'Data dimensions must match existing dimensions in dataset'); %新数据维度和已存在的数据相同。
     
      assert(prev_lab_sz(1:end-1)==lab_dims(1:end-1), 'Label dimensions must match existing dimensions in dataset');
      startloc.dat=[ones(1,length(dat_dims)-1), prev_dat_sz(end)+1];  %加在之前的数据后面，从原数据后一个初始化全1矩阵
     
      startloc.lab=[ones(1,length(lab_dims)-1), prev_lab_sz(end)+1];
    end
  end

  if ~isempty(data)
    h5write(filename, '/data', single(data), startloc.dat, size(data));%将数据写入HDF5数据集，从startloc.dat为开始写入的索引位置
   
    h5write(filename, '/label', single(labels), startloc.lab, size(labels));  
  end

  if nargout   %nargout返回函数输出参数的数量
    info=h5info(filename);      %返回有关 filename 指定的整个 HDF5 文件的信息
    curr_dat_sz=info.Datasets(1).Dataspace.Size;
  
    curr_lab_sz=info.Datasets(2).Dataspace.Size;
  end
end
