 function kdata = readframes(pfile, frames, varargin)
%function kdata = readframes(pfile, frames, 'option name', option, ...)
%|
%| Read in k-space data from P-files (using the TOPPE library) 
%| acquired on GE scanners.
%|
%| Input:
%|   pfile : String containing path to P-file
%|   frames : 1D array containing frame indices to be read from
%|			  the P-file
%|
%| Options:
%|	 'istart' : Start index; this decides how many samples to
%|				throw away at start of readout (default: 1)
%|	 'istop'  : End index; this decides how many samples to
%|				throw away at end of readout (default: number of fid points)
%|	 'nrot'   : Number of spiral arms per kz-encode (default: 3)
%| 	 'outfile': Path to a .h5 file if k-space data is required to be saved
%|				to file. (Default: "", which does not save to file)
%|	 'z_undersample_list' : 1D array containing kz-encodes to perform retrospective undersampling.
%|				(Default: [], which does not perform retrospective undersampling)
%|   'spiral_undersample_list' : 1D array containing which spiral arms to pick in each
%| 				kz-encode. (Default: [], which corresponds to no retrospective undersampling)
%|
%| Output:
%|	 kdata : Array of size [nframes, ncoil, nshot, nread] containing complex k-space data
%|			 for the 3D stack-of-spirals acquisition
%|
%| Example:
%|	 Suppose we want to read data for frames 2 and 5, from a given P-file by throwing away the first 100
%|	 samples, and keeping the next 3000 samples. Also, say we want to perform retrospective
%|	 undersampling factor with an undersampling factor of 2 along kz, and factor nrot in-plane. Then,
%|
%|	 z_undersample_list = 1:nz/2;
%| 	 spiral_undersample_list = [repmat(1:nrot, 1, floor(nz/2 / nrot)), 1:mod(nz/2, nrot)];
%|   kdata = readframes('path/to/P-file', [2,5], 'istart', 101, 'istop', 3101, 
%|           'z_undersample_list', z_undersample_list, 'spiral_undersample_list', spiral_undersample_list);

	% read spiral kspace data from p-file
	d = toppe.utils.loadpfile(pfile); % [nfid, ncoil, nz, necho, nview]
	d = flip(d, 1);
    [nfid, ncoil, nz, necho, nview] = size(d);

	% default options
	arg.istart = 1; % indices for readout
	arg.istop = nfid;
	arg.nrot = 3; % number of spiral rotations per kz-encode
	arg.z_undersample_list = []; % default is to not perform retrospective undersampling
	arg.spiral_undersample_list = [];
	arg.outfile = ""; % default is to not save to file

	arg = vararg_pair(arg, varargin); % from mirt toolbox

	% check arguments for undersampling
	assert(length(arg.z_undersample_list) == length(arg.spiral_undersample_list),...
	 "For undersampling, number of kz-encodes should match the number of spiral arms.")

	% extract views corresponding to the desired frames
	nread = arg.istop - arg.istart + 1;
	nframes = length(frames);
	frames = reshape(frames, 1, []); % row vector
	views = (frames-1) * arg.nrot + [1:arg.nrot]'; % [nrot, nframes]
	d = squeeze(d(arg.istart:arg.istop, :, :, 1, views(:))); % [nread, ncoil, nz, nviews]

	% reshape and permute to obtain the required order of dimensions
	d = reshape(d, [nread, ncoil, nz, arg.nrot, nframes]); % [nread, ncoil, nz, nrot, nframes]
	d = permute(d, [5 2 3 4 1]); % [nframes, ncoil, nz, nrot, nread]

	% retrospective undersampling
	if isequal(arg.z_undersample_list, [])
		d = permute(d, [1, 2, 4, 3, 5]); % [nframes, ncoil, nrot, nz, nread]
		kdata = reshape(d, [nframes, ncoil, nz * arg.nrot, nread]); % [nframes, ncoil, nshot, nread]
	else
		nshot = length(arg.spiral_undersample_list); % number of shots in the undersampled dataset
		kdata = zeros(nframes, ncoil, nshot, nread); % [nframes, ncoil, nshot, nread]
		for ii = 1:nshot
			kdata(:,:,ii,:) = d(:, :, arg.z_undersample_list(ii), arg.spiral_undersample_list(ii), :);
		end
	end

	if arg.outfile ~= ""

		% save to file
		[filepath, filename, fileext] = fileparts(arg.outfile);

		if fileext == ".h5"
			if isfile(arg.outfile)
				delete(arg.outfile)
			end

			% create output file
			h5create(arg.outfile, '/kdata_r', size(kdata))
			h5create(arg.outfile, '/kdata_i', size(kdata))

			% write data to file
			disp('Writing kspace data to file...')
			h5write(arg.outfile, '/kdata_r', real(kdata));
			h5write(arg.outfile, '/kdata_i', imag(kdata));
			disp('Done.')
		else
			warning(sprintf('Only h5 files are supported currently. Unable to save kspace data to file: %s', arg.outfile))
		end

	end
end