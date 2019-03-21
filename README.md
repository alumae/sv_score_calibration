# Score calibrator for speaker verification

This software allows you to calibrate log likelihood ratio (LLR) scores 
for speaker verification evaluation.

Often, speaker verification systems are evaluated on the actual DCF scores
or CLLR scores. In order to get good performance on these measures, LLR
scores from a speaker verification system (e.g., from a PLDA model) need
to be calibrated.

This software allows one to optimize the CLLR measure of a speaker 
verification system (or a combination of systems). 
Calibration is done by finding a linear transform
that optimizes the CLLR measure of the heldout data. 

## Requirements

  * Python 3+
  * Pytorch 1.0 (no GPU needed)

## Example

In the `samples`  directory, there are thee files: `sys1_llr.txt`,
`sys2_llr.txt` and `trial-keys.txt`. The first two are (uncalibrated)
LLR scores from two different systems on a heldout trial set, and the third file
gives the oracle values for all trials -- target (`tgt`) or non-target/impostor (`imp`).

In order to measure the DCF and CLLR scores, you need to download and extract a scoring tool
from https://app.box.com/s/9tpuuycgxk9hykr6romsv05vvmdpie11/file/389271165078.
It's an official scorer for the [The VOiCES from a Distance Challenge](https://voices18.github.io/Interspeech2019_SpecialSession/).

Let's first measure the accuracy of the uncalibrated scores:

    $ python2 voices_scorer/score_voices sample/sys1_llr.txt sample/trial-keys.txt                 
    minDCF   : 0.4252
    actDCF   : 0.7496
    avgRPrec : 0.6384
    EER      : 0.0547
    Cllr     : 0.9787

    $ python2 /export/home/tanel/data/IS2019_VOiCES/Development_Data/Speaker_Recognition/voices_scorer/score_voices sample/sys2_llr.txt sample/trial-keys.txt                 
    minDCF   : 0.3034
    actDCF   : 1.8849
    avgRPrec : 0.6979
    EER      : 0.0710
    Cllr     : 0.5986

The software can calibrate scores of one or more systems. Let's first try
to calibrate the 1st system (`sample/sys2_llr.txt`). First, you have to find
the parameters that optimize CLLR of heldout data:

    $ python calibrate_scores.py --save-model sample/sys1_calibration.pth sample/trial-keys.txt sample/sys1_llr.txt                                                           
    Starting point for CLLR is 0.978737
    STEP:  0
      loss: 0.5246010594024472
      [...]
      loss: 0.18544731777635964
    Converged!
    Saving model to sample/sys1_calibration.pth
    
Next, you need to *apply* the calibration model:

    $ python apply_calibration.py sample/sys1_calibration.pth sample/sys1_llr.txt sample/sys1_calibrated_llr.txt
    
Let's measure the performance of the calibrated system:

    $ python2 voices_scorer/score_voices sample/sys1_calibrated_llr.txt sample/trial-keys.txt      
    minDCF   : 0.4252
    actDCF   : 0.4320
    avgRPrec : 0.6384
    EER      : 0.0547
    Cllr     : 0.1854

As can be seen, the `actDCF` and `Cllr` scores are now much better than initially.

You can also calibrate a fusion of two or more systems:

    $ python calibrate_scores.py --save-model sample/sys1_sys2_calibration.pth sample/trial-keys.txt sample/sys1_llr.txt sample/sys2_llr.txt                                 
    Starting point for CLLR is 0.788658
    STEP:  0
      loss: 0.711224191738577
      loss: 0.7045015511238044
      [...]
      loss: 0.18383203478911536
      loss: 0.18382984498508542
    Converged!
    Saving model to sample/sys1_sys2_calibration.pth

Apply the model:

    $ python apply_calibration.py sample/sys1_sys2_calibration.pth sample/sys1_llr.txt sample/sys2_llr.txt sample/sys1_sys2_calibrated_llr.txt 
    
Measure the performance:

    $ python2 voices_scorer/score_voices sample/sys1_sys2_calibrated_llr.txt sample/trial-keys.txt 
     
    minDCF   : 0.3516
    actDCF   : 0.3586
    avgRPrec : 0.6592
    EER      : 0.0533
    Cllr     : 0.1838


You can cite the following paper if you use the software in research:

    @inproceedings{alumae2019taltech,
      author={Tanel Alum\"{a}e, Asadullah},
      title={The {TalTech} Systems for the {VOiCES from a Distance Challenge}},
      year=2019,
      booktitle={Interspeech (submitted)},
    }
