#! /usr/bin/env python

'''
Converts raw HRTF databases into this repo's custom hdf5 schema (as read by
utilities/read_hdf5.py), producing <db_path>/<db>.hdf5. This is the missing
"<name>_to_hdf5.py" step referenced in src/README.md.

Usage:
    python convert_to_hdf5.py cipic all -f
    python convert_to_hdf5.py cipic 003 008 -f

Adding a new database means writing one adapter class (see CipicAdapter) and
registering it in DATABASES below. Only 'cipic' is implemented.
'''

import argparse
import glob
import os
import re

import h5py
import numpy as np
import scipy.io as sio


# Canonical CIPIC measurement grid (25 azimuths x 50 elevations, 1250 total).
# Used only as a self-check that the converted positions land on the expected
# grid -- not written to the file, not used by any other part of the pipeline.
CIPIC_AZIMUTHS = np.array(
    [-80, -65, -55, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0,
     5, 10, 15, 20, 25, 30, 35, 40, 45, 55, 65, 80], dtype=np.float64)
CIPIC_ELEVATIONS = -45.0 + 5.625 * np.arange(50)
GRID_SNAP_TOLERANCE = 1e-6  # degrees; measured actual deviation is ~1e-9


class CipicAdapter:
    name = 'cipic'
    default_src = 'cipic_sofa'

    def __init__(self, src_dir):
        self.src_dir = src_dir
        self._anthro = None  # lazy-loaded {id (int): row index}

    def _sofa_path(self, sid):
        return os.path.join(self.src_dir, 'subject_%s.sofa' % sid)

    def list_subjects(self):
        paths = glob.glob(os.path.join(self.src_dir, 'subject_*.sofa'))
        ids = []
        for p in paths:
            m = re.search(r'subject_(\d+)\.sofa$', os.path.basename(p))
            if m:
                ids.append(m.group(1))
        return sorted(ids, key=int)

    def load_subject(self, sid):
        path = self._sofa_path(sid)
        if not os.path.isfile(path):
            raise FileNotFoundError("No .sofa file for subject '%s' at %s" % (sid, path))

        with h5py.File(path, 'r') as f:
            ir = f['Data.IR'][:]              # (npos, 2, ntaps) -- receiver 0=left, 1=right
            sofa_pos = f['SourcePosition'][:]  # (npos, 3) SOFA spherical degrees
            fs = float(np.asarray(f['Data.SamplingRate'][:]).flatten()[0])

        hrir_l = ir[:, 0, :]
        hrir_r = ir[:, 1, :]
        srcpos = self._to_interaural_polar(sofa_pos)

        return {'hrir_l': hrir_l, 'hrir_r': hrir_r, 'srcpos': srcpos, 'fs': fs}

    def _to_interaural_polar(self, sofa_pos):
        '''
        SOFA spherical (az, el, r) in degrees -> cartesian -> CIPIC interaural-polar
        (lateral theta, polar phi, r). See plan for why this convention is used
        instead of storing SOFA angles verbatim.
        '''
        az = np.radians(sofa_pos[:, 0])
        el = np.radians(sofa_pos[:, 1])
        r = sofa_pos[:, 2]

        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)

        theta = np.degrees(np.arcsin(np.clip(y / r, -1.0, 1.0)))
        phi = np.degrees(np.arctan2(z, x))
        # Wrap into [-45, 315). The epsilon guards the -45 boundary: without it,
        # values that should land exactly on -45 arrive as -45.0000000001 and
        # get thrown to ~315 by the modulo (verified bug, see plan).
        phi = (phi + 45.0 + 1e-9) % 360.0 - 45.0

        theta_snapped, theta_dev = self._snap(theta, CIPIC_AZIMUTHS)
        phi_snapped, phi_dev = self._snap(phi, CIPIC_ELEVATIONS)

        max_dev = max(theta_dev, phi_dev)
        if max_dev > GRID_SNAP_TOLERANCE:
            raise ValueError(
                "Measured position grid does not match CIPIC's canonical 25x50 grid "
                "(max deviation %.6f deg, tolerance %.6f). Refusing to snap silently."
                % (max_dev, GRID_SNAP_TOLERANCE))

        return np.stack([theta_snapped, phi_snapped, r], axis=1)

    @staticmethod
    def _snap(values, grid):
        idx = np.argmin(np.abs(values[:, None] - grid[None, :]), axis=1)
        snapped = grid[idx]
        deviation = np.abs(values - snapped).max()
        return snapped, deviation

    def load_anthro(self, sid):
        if self._anthro is None:
            path = os.path.join(self.src_dir, 'anthro.mat')
            if not os.path.isfile(path):
                raise FileNotFoundError("No anthro.mat found at %s" % path)
            mat = sio.loadmat(path)
            ids = mat['id'].flatten().astype(int)
            self._anthro = {'ids': ids, 'mat': mat, 'index': {v: i for i, v in enumerate(ids)}}

        idx_map = self._anthro['index']
        mat = self._anthro['mat']
        sid_int = int(sid)
        if sid_int not in idx_map:
            raise KeyError(
                "Subject '%s' has a .sofa file but no matching row in anthro.mat "
                "(no id == %d)" % (sid, sid_int))
        row = idx_map[sid_int]

        return {
            'X': mat['X'][row].astype(np.float64),
            'D': mat['D'][row].astype(np.float64),
            'theta': mat['theta'][row].astype(np.float64),
            'age': float(mat['age'][row][0]),
            'WeightKilograms': float(mat['WeightKilograms'][row][0]),
            'id': float(sid_int),
            'sex': str(mat['sex'][row]),
        }


DATABASES = {'cipic': CipicAdapter}


def convert(db, adapter, subjects, out_path, force):
    nan_subjects = []

    with h5py.File(out_path, 'a') as f:
        for sid in subjects:
            gname = 'subject_%s' % sid
            if gname in f:
                if not force:
                    print("Skipping %s (already exists in %s; use -f to overwrite)"
                          % (gname, out_path))
                    continue
                del f[gname]

            subj_data = adapter.load_subject(sid)
            npos = subj_data['srcpos'].shape[0]

            g = f.create_group(gname)
            g.create_dataset('hrir_l/raw', data=subj_data['hrir_l'])
            g.create_dataset('hrir_r/raw', data=subj_data['hrir_r'])
            g.create_dataset('srcpos/raw', data=subj_data['srcpos'])
            g.create_dataset('nn/raw', data=np.zeros((npos, 1), dtype=np.float64))
            g.attrs['fs'] = np.array([subj_data['fs']], dtype=np.float64)

            anthro = adapter.load_anthro(sid)
            for key, value in anthro.items():
                if key == 'sex':
                    g.attrs[key] = value
                else:
                    g.attrs[key] = np.asarray(value, dtype=np.float64).ravel()

            if np.isnan(anthro['X']).any() or np.isnan(anthro['D']).any() or np.isnan(anthro['theta']).any():
                nan_subjects.append(sid)

            print("Wrote %s (%d positions)" % (gname, npos))

    if nan_subjects:
        print("WARNING: NaN anthropometry values for subjects: %s"
              % ', '.join(nan_subjects))


def parseargs():
    parser = argparse.ArgumentParser(
        description='Convert a raw HRTF database into this repo\'s custom hdf5 schema.')
    parser.add_argument('db', type=str, help='Database name (e.g. cipic)')
    parser.add_argument('subjects', type=str, nargs='+',
                         help="Subject id(s), or 'all'")
    parser.add_argument('-d', '--dir', dest='directory', type=str, default='../../datasets/',
                         help='Output directory; <db>.hdf5 is written here (default: ../../datasets/)')
    parser.add_argument('-s', '--src', dest='src', type=str, default=None,
                         help='Source directory containing the raw database files '
                              '(default: <dir>/<adapter default_src>)')
    parser.add_argument('-f', '--force', dest='force', action='store_true',
                         help='Overwrite existing subject groups in the output file')
    parser.add_argument('-list', dest='list_subjects', action='store_true',
                         help='Print available subject ids for this database and exit')
    return vars(parser.parse_args())


def main():
    args = parseargs()

    if args['db'] not in DATABASES:
        raise ValueError("Unknown database '%s'. Available: %s"
                          % (args['db'], ', '.join(sorted(DATABASES.keys()))))

    adapter_cls = DATABASES[args['db']]
    src_dir = args['src'] if args['src'] is not None else os.path.join(args['directory'], adapter_cls.default_src)
    adapter = adapter_cls(src_dir)

    if args['list_subjects']:
        print('\n'.join(adapter.list_subjects()))
        return

    if 'all' in args['subjects']:
        subjects = adapter.list_subjects()
    else:
        subjects = args['subjects']
        available = set(adapter.list_subjects())
        missing = [s for s in subjects if s not in available]
        if missing:
            raise ValueError("Requested subject(s) not found in %s: %s"
                              % (src_dir, ', '.join(missing)))

    out_path = os.path.join(args['directory'], args['db'] + '.hdf5')
    convert(args['db'], adapter, subjects, out_path, args['force'])
    print("Done. Wrote %d subject(s) to %s" % (len(subjects), out_path))


if __name__ == '__main__':
    main()
