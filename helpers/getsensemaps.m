function smaps = getsensemaps(kdata, varargin)
%function smaps = getsensemaps(kdata, 'option name', option, ...)
%|
%| Estimate sensitivity maps for parallel imaging acquisitions using the BART toolbox.
%|
%| Input:
%|   kdata : Array of size [nx, ny, nz, ncoils] representing multi-coil k-space
%| 			 from which to estimate sensitivity maps.
%| Options:
%| 	 'outfile': Path to a .h5 file if k-space data is required to be saved
%|				to file. (Default: "", which does not save to file)
%|
%| Output:
%|	 smaps : Array of size [nx, ny, nz, ncoil] containing complex-valued sensitivity maps.

	arg.outfile = ""; % default is to not save to file

	arg = vararg_pair(arg, varargin); % from mirt toolbox

	% compute sense maps
	disp('Computing sensitivity maps using BART...')
	tic; sens_bart = bart('ecalib -r 20', kdata); toc;
	smaps = sens_bart(:,:,:,:,1); % [nx, ny, nz, ncoil]
	smaps = permute(smaps, [4,1,2,3]); % [ncoil, nx. ny, nz]
	disp('Done.')

	if arg.outfile ~= ""

		% save to file
		[filepath, filename, fileext] = fileparts(arg.outfile);

		if fileext == ".h5"
			if isfile(arg.outfile)
				delete(arg.outfile)
			end

			% create output file
			h5create(arg.outfile, '/smaps_r', size(smaps))
			h5create(arg.outfile, '/smaps_i', size(smaps))

			% write data to file
			disp('Writing sensitivity maps to file...')
			h5write(arg.outfile, '/smaps_r', real(smaps));
			h5write(arg.outfile, '/smaps_i', imag(smaps));
			disp('Done.')
		else
			warning(sprintf('Only h5 files are supported currently. Unable to save sensitivity maps to file: %s', arg.outfile))
		end

	end

end