import os
import h5py
from sqlitedict import SqliteDict
import numpy as np
import re
# aveable upon request (Arthur Zalevsky)
import extract_result_slim as er 
import prody
import argparse as ag


class PDBGenerator():
    def __init__(self, db, resfile, outdir, inputfile):
        self.db = SqliteDict(db)
        self.resfile = h5py.File(resfile)
        self.outdir = outdir
        self.inputfile = inputfile
        if not self.outdir.endswith('/'):
            self.outdir += '/'

        def makePepCoord(self, peptides: list, overlaps: list):
            # number_of_pep = len(peptides)
            last_pep = peptides[0][:-1]
            last_conform = peptides[0][-1]
            db_current_pep = self.db[last_pep]
            current_pep_ind = db_current_pep.getResindices()
            current_pep_decap = np.where(
                np.logical_and(current_pep_ind != 0,
                               current_pep_ind <= len(last_pep)))[0]
            coord = np.array(
                self.resfile[last_pep][last_conform][current_pep_decap, :])

            for i in range(1, len(peptides)):
                last_pep = peptides[i][:-1]
                last_conform = peptides[i][-1]
                last_overlap = overlaps[i - 1]

                db_current_pep = self.db[last_pep]
                current_pep_ind = db_current_pep.getResindices()
                current_pep_decap = np.where(
                    np.logical_and(current_pep_ind > last_overlap,
                                   current_pep_ind <= len(last_pep)))[0]

                coord = np.vstack([
                    coord,
                    np.array(self.resfile[last_pep][last_conform]
                             [current_pep_decap, :])
                ])

            return coord

        def parsePathFile(self):
            with open(self.inputfile, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    massives = re.findall('\[.*\]', line)
                    massives = massives[0].split('],')
                    peps = massives[0].replace("[", "").replace(
                        "]", "").replace(" ", "").replace("'", "").split(",")
                    overlaps = [
                        int(i) for i in massives[1].replace("[", "").replace(
                            " ", "").replace("]", "").split(",")
                    ]
                    yield peps, overlaps

        def PDB_creator(self, peptides: list, overlaps: list):

            res = er.PepExtractor(database=self.db, resfile=self.resfile)

            last_pep = peptides[0][:-1]
            last_conform = peptides[0][-1]
            r1 = res.extract_result(str(last_pep), int(last_conform))
            seq = last_pep
            for i in range(1, len(peptides)):
                last_pep = peptides[i][:-1]
                last_conform = peptides[i][-1]
                last_overlap = overlaps[i - 1]
                r2 = res.extract_result(str(last_pep), int(last_conform))
                r1_ = r1.select(
                    'resnum 2:{0}'.format(len(seq) + 2)).toAtomGroup()
                r2_ = r2.select('resnum ' + str(last_overlap + 2) +
                                ':5').toAtomGroup()
                c_ = 5
                for r_ in r2_.iterResidues():
                    r_.setResnum(c_)
                    c_ += 1
                r1 = r1_ + r2_
                seq += last_pep[last_overlap:]

                prody.writePDB(
                    '{0}{1}{2}.pdb'.format(self.outdir, str(len(seq)), seq),
                    r1)

        def generatePDB(self):
            for pep, over in self.parsePathFile():
                self.PDB_creator(pep, over)


def arg_pars():
    parser = ag.ArgumentParser()

    parser.add_argument(
        "-i",
        dest="--input-file",
        required=True,
        type=str,
        help="File with petides lists and their overlaps")
    parser.add_argument(
        "-db", dest="db", required=True, type=str, help="Database")
    parser.add_argument(
        "-resfile",
        dest="resfile",
        required=True,
        type=str,
        help="file with reference coordinates in hdf format")
    parser.add_argument(
        "-o",
        "--out",
        dest="outdir",
        type=str,
        help="Out directory",
        default=os.getcwd())

    # parser.add_argument("-o","--out", type=str,
    # help="result stdout", default=sys.stdout())

    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':
    args = arg_pars()
    jn = PDBGenerator(**args)
    jn.generatePDB()
