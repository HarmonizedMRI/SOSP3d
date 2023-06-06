function ktraj = getspiraltrajs(modfile, fov, N, varargin)
%function ktraj = getspiraltrajs(modfile, fov, N, 'option name', option, ...)
%|
%| Obtain k-space locations for the given 3D stack-of-spirals acquisition.
%|
%| Input:
%|   modfile : String containing path to readout.mod file for the 
%|			   3d stack-of-spirals acquisition
%|   fov : 1D array containing field of view information in all
%|         three dimensions (units of cm or mm)
%| 	 N : 1D array containing matrix size of acquisition
%|
%| Options:
%|	 'istart' : Start index; this decides how many samples to
%|				throw away at start of readout (default: 1)
%|	 'istop'  : End index; this decides how many samples to
%|				throw away at end of readout (default: number of fid points)
%|	 'dwell_time' : Dwell time or sampling interval in milliseconds (Default: 4e-3)
%| 	 'outfile': Path to a .h5 file if k-space data is required to be saved
%|				to file. (Default: "", which does not save to file)
%|	 'z_undersample_list' : 1D array containing kz-encodes to perform retrospective undersampling.
%|				(Default: [], which does not perform retrospective undersampling)
%|   'spiral_undersample_list' : 1D array containing which spiral arms to pick in each
%| 				kz-encode. (Default: [], which corresponds to no retrospective undersampling)
%|
%| Output:
%|	 ktraj : Array of size [3, nshot, nread] containing k-space locations for the 
%|   		 3d stack-of-spirals trajectory. These are in units of radians (must be
%|			 between -pi and pi). 
%|
%| Example:
%|	 fov = [22, 22, 10]; % in cm
%|   N = [92, 92, 50]; % matrix size
%|	 z_undersample_list = 1:nz/2;
%| 	 spiral_undersample_list = [repmat(1:nrot, 1, floor(nz/2 / nrot)), 1:mod(nz/2, nrot)];
%|   ktraj = getspiraltrajs('path/to/readout.mod', fov, N, 'istart', 101, 'istop', 3101, 
%|           'z_undersample_list', z_undersample_list, 'spiral_undersample_list', spiral_undersample_list);

	% setup
	gamma = 4.2576;      % kHz/Gauss

	% read .mod file to get gradient information
	[~, gx, gy, gz] = toppe.readmod(modfile); 

	% default options
	arg.istart = 1;
	arg.istop = size(gx, 1); % default value is to consider all readout points
	arg.dwell_time = 4e-3; % in milliseconds
	arg.outfile = ""; % default is to not save to file
	arg.z_undersample_list = []; % default is to not perform retrospective undersampling
	arg.spiral_undersample_list = [];

	arg = vararg_pair(arg, varargin);

	% check arguments for undersampling
	assert(length(arg.z_undersample_list) == length(arg.spiral_undersample_list),...
	 "For undersampling, number of kz-encodes should match the number of spiral arms.")

	% compute kspace spiral trajectories
	kx = gamma * arg.dwell_time * cumsum(gx);   % cycles/unit distance
	ky = gamma * arg.dwell_time * cumsum(gy);   % cycles/unit distance
	kx = kx(arg.istart:arg.istop, :);
	ky = ky(arg.istart:arg.istop, :);

	% compute kspace locations for the kz-encodes (which follow a Cartesian structure along z in 
	% stack-of-spirals acquisitions)
	nz = N(3);
	fovz = fov(3);
	kzmax = 0.5 / (fovz/nz);
	kz = linspace(-kzmax + 1/fovz/2, kzmax - 1/fovz/2, nz);

	% create 3-dimensional kspace trajectories
	[nread, nrot] = size(kx);
	spiral_trajs = zeros(3, nz, nrot, nread); % [3, nz, nrot, nread]
	for iz = 1:nz
		spiral_trajs(1, iz, :, :) = kx';
		spiral_trajs(2, iz, :, :) = ky';
		spiral_trajs(3, iz, :, :) = kz(iz);
	end

	% retrospective undersampling
	if isequal(arg.z_undersample_list, [])
		ktraj = permute(spiral_trajs, [1,3,2,4]); % [3, nrot, nz, nread]
		ktraj = reshape(ktraj, [3, nrot*nz, nread]); % [3, nshot, nread]
	else
		nshot = length(arg.spiral_undersample_list); % number of spiral arms in the undersampled dataset
		ktraj = zeros(3, nshot, nread);
		for ii = 1:nshot
			ktraj(:, ii, :) = spiral_trajs(:, arg.z_undersample_list(ii), arg.spiral_undersample_list(ii), :); % [3, nshot, nread]
		end
	end

	% convert from units of cycles/cm or cycles/mm to radians
	for d = 1:3
		ktraj(d,:,:) = ktraj(d,:,:) * 2 * pi * fov(d) / N(d);
	end

	% save kspace trajectories to file?
	if arg.outfile ~= ""

		% save to file
		[filepath, filename, fileext] = fileparts(arg.outfile);

		if fileext == ".h5"
			if isfile(arg.outfile)
				delete(arg.outfile)
			end

			% create output file
			h5create(arg.outfile, '/ktraj', size(ktraj))
			h5create(arg.outfile, '/N', size(N))
			h5create(arg.outfile, '/fov', size(fov))

			% write data to file
			disp('Writing spiral trajectory data to file...')
			h5write(arg.outfile, '/ktraj', ktraj);
			h5write(arg.outfile, '/N', N);
			h5write(arg.outfile, '/fov', fov);
			disp('Done.')
		else
			warning(sprintf('Only h5 files are supported currently. Unable to save spiral trajectory data to file: %s', arg.outfile))
		end

	end
end