"""
Bruker data reading functionality for the Enhanced Bruker NMR Data Reader
"""

import os
import numpy as np
import nmrglue as ng
from utils.helpers import get_dimension

class BrukerDataReader:
    def __init__(self):
        pass

    def find_experiments(self, base_folder):
        """Find all experiment folders in the base directory"""
        experiment_paths = []
        
        # Find all experiment folders (those containing pdata subdirectories)
        for item in os.listdir(base_folder):
            exp_path = os.path.join(base_folder, item)
            if os.path.isdir(exp_path) and item.isdigit():
                pdata_path = os.path.join(exp_path, "pdata")
                if os.path.exists(pdata_path):
                    # Check what dimensions are available in this experiment
                    dimensions = set()
                    proc_folders = []
                    
                    for proc_item in os.listdir(pdata_path):
                        proc_path = os.path.join(pdata_path, proc_item)
                        if os.path.isdir(proc_path) and proc_item.isdigit():
                            proc_folders.append(proc_item)
                            try:
                                dic, _ = ng.bruker.read_pdata(proc_path)
                                dim = get_dimension(dic)
                                dimensions.add(dim)
                            except:
                                dimensions.add(0)
                    
                    if proc_folders:
                        dim_str = "/".join([f"{d}D" for d in sorted(dimensions) if d > 0])
                        display_name = f"Exp {item} ({dim_str}) - {len(proc_folders)} procno(s)"
                        experiment_paths.append((exp_path, dimensions, display_name))
        
        return experiment_paths

    def read_experiments(self, selected_paths, matrices_dict, progress_bar, root):
        """Read selected experiments and populate matrices_dict"""
        matrices_dict.clear()
        
        # Count total pdata folders to read
        total_pdata = 0
        for exp_path in selected_paths:
            pdata_path = os.path.join(exp_path, "pdata")
            if os.path.exists(pdata_path):
                for proc_item in os.listdir(pdata_path):
                    proc_path = os.path.join(pdata_path, proc_item)
                    if os.path.isdir(proc_path) and proc_item.isdigit():
                        total_pdata += 1
        
        progress_bar['maximum'] = total_pdata
        progress_counter = 0
        
        for exp_path in selected_paths:
            expno = os.path.basename(exp_path)
            pdata_path = os.path.join(exp_path, "pdata")
            
            if not os.path.exists(pdata_path):
                continue
                
            for proc_item in os.listdir(pdata_path):
                proc_path = os.path.join(pdata_path, proc_item)
                if os.path.isdir(proc_path) and proc_item.isdigit():
                    procno = proc_item
                    
                    progress_bar['value'] = progress_counter
                    root.update_idletasks()
                    progress_counter += 1
                    
                    try:
                        dic, data = ng.bruker.read_pdata(proc_path)
                        
                        # Get proper dimensions using Bruker parameters
                        matrix = self.get_proper_dimensions(data, dic)
                        
                        # Store both data and dictionary
                        if expno not in matrices_dict:
                            matrices_dict[expno] = {}
                            
                        matrices_dict[expno][procno] = {
                            'data': matrix,
                            'dic': dic,
                            'path': proc_path
                        }

                        print(f"Loaded {proc_path}: {matrix.shape}, {matrix.dtype}")
                        
                    except Exception as e:
                        print(f"Failed to read {proc_path}: {e}")
                        continue

        progress_bar['value'] = total_pdata
        return True

    def get_proper_dimensions(self, data, dic):
        """Get properly dimensioned data using Bruker SI parameters"""
        matrix = np.array(data)
        
        # Always preserve original SI dimensions from dictionary
        if matrix.ndim == 1:
            # For 1D data, get SI from procs
            if 'procs' in dic and 'SI' in dic['procs']:
                target_si = int(dic['procs']['SI'])
                if matrix.shape[0] != target_si:
                    # Pad or truncate to match SI
                    if matrix.shape[0] > target_si:
                        matrix = matrix[:target_si]
                    else:
                        padded = np.zeros(target_si, dtype=matrix.dtype)
                        padded[:matrix.shape[0]] = matrix
                        matrix = padded
        
        elif matrix.ndim == 2:
            # For 2D data, get SI from both dimensions
            target_si_f1 = matrix.shape[0]  # Default to current
            target_si_f2 = matrix.shape[1]  # Default to current
            
            if 'proc2s' in dic and 'SI' in dic['proc2s']:
                target_si_f1 = int(dic['proc2s']['SI'])
            if 'procs' in dic and 'SI' in dic['procs']:
                target_si_f2 = int(dic['procs']['SI'])
            
            # Ensure dimensions match SI parameters exactly
            if matrix.shape != (target_si_f1, target_si_f2):
                new_matrix = np.zeros((target_si_f1, target_si_f2), dtype=matrix.dtype)
                # Copy overlapping region
                copy_f1 = min(matrix.shape[0], target_si_f1)
                copy_f2 = min(matrix.shape[1], target_si_f2)
                new_matrix[:copy_f1, :copy_f2] = matrix[:copy_f1, :copy_f2]
                matrix = new_matrix
        
        return matrix

    def get_axis_ranges(self, dic, dimension):
        """Calculate proper axis ranges for plotting"""
        if dimension == 1:
            # 1D spectrum - F1 dimension
            if 'procs' in dic:
                procs = dic['procs']
                si = procs.get('SI', 1024)
                sw_h = procs.get('SW_h', 10000.0)  # Hz
                o1 = procs.get('OFFSET', 0.0)  # ppm
                sf = procs.get('SF', 400.0)  # MHz
                
                # Create ppm axis
                ppm_range = sw_h / sf  # Convert Hz to ppm
                ppm_axis = np.linspace(o1, o1 - ppm_range, si)
                return ppm_axis
        
        elif dimension == 2:
            # 2D spectrum - F1 and F2 dimensions
            f2_axis = f1_axis = None
            
            if 'procs' in dic:  # F2 dimension (direct)
                procs = dic['procs']
                si_f2 = procs.get('SI', 1024)
                sw_h_f2 = procs.get('SW_h', 10000.0)
                o1_f2 = procs.get('OFFSET', 0.0)
                sf_f2 = procs.get('SF', 400.0)
                
                ppm_range_f2 = sw_h_f2 / sf_f2
                f2_axis = np.linspace(o1_f2, o1_f2 - ppm_range_f2, si_f2)
            
            if 'proc2s' in dic:  # F1 dimension (indirect)
                proc2s = dic['proc2s']
                si_f1 = proc2s.get('SI', 256)
                sw_h_f1 = proc2s.get('SW_h', 10000.0)
                o1_f1 = proc2s.get('OFFSET', 0.0)
                sf_f1 = proc2s.get('SF', 400.0)
                
                ppm_range_f1 = sw_h_f1 / sf_f1
                f1_axis = np.linspace(o1_f1, o1_f1 - ppm_range_f1, si_f1)
            
            return f1_axis, f2_axis
        
        return None

    def check_si_compatibility(self, spec1, spec2, dic1, dic2):
        """Check if two spectra have compatible SI parameters"""
        if spec1.ndim != spec2.ndim:
            return False
            
        if spec1.ndim == 2:
            # Check F1 and F2 SI parameters
            si_f1_1 = dic1.get('proc2s', {}).get('SI', spec1.shape[0])
            si_f2_1 = dic1.get('procs', {}).get('SI', spec1.shape[1])
            si_f1_2 = dic2.get('proc2s', {}).get('SI', spec2.shape[0])
            si_f2_2 = dic2.get('procs', {}).get('SI', spec2.shape[1])
            
            return (si_f1_1 == si_f1_2) and (si_f2_1 == si_f2_2)
        
        return spec1.shape == spec2.shape

    def export_metadata(self, matrices_dict, filename):
        """Export metadata of all loaded spectra"""
        try:
            with open(filename, 'w') as f:
                f.write("Bruker NMR Data Metadata\n")
                f.write("=" * 50 + "\n\n")
                
                for expno, procnos in matrices_dict.items():
                    f.write(f"Experiment: {expno}\n")
                    f.write("-" * 30 + "\n")
                    
                    for procno, data_info in procnos.items():
                        matrix = data_info['data']
                        dic = data_info['dic']
                        path = data_info['path']
                        
                        f.write(f"  Processing Number: {procno}\n")
                        f.write(f"  Path: {path}\n")
                        f.write(f"  Shape: {matrix.shape}\n")
                        f.write(f"  Data Type: {matrix.dtype}\n")
                        f.write(f"  Dimensions: {matrix.ndim}D\n")
                        
                        if matrix.ndim == 1:
                            f.write(f"  Min/Max (real): {matrix.real.min():.3e}/{matrix.real.max():.3e}\n")
                        elif matrix.ndim == 2:
                            f.write(f"  Min/Max (real): {matrix.real.min():.3e}/{matrix.real.max():.3e}\n")
                        
                        # Add some key parameters from dictionary if available
                        if 'procs' in dic and 'SI' in dic['procs']:
                            f.write(f"  Processing Size: {dic['procs']['SI']}\n")
                        
                        f.write("\n")
                    f.write("\n")
            
            return True
        except Exception as e:
            print(f"Failed to export metadata: {e}")
            return False