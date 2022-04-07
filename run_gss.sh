
# . ./utils/parse_options.sh

# . ./cmd.sh
# . ./path.sh
source ./bashrc

set -e
stage=0
nj=1
type=train
# path settings
data_root=/raw7/cv1/hangchen2/misp2021_avsr/feature/misp2021_avsr/${type}_far_audio_wpe    #eg. ./misp2021
python_path=~/anaconda3/bin/ #eg. ./bin
out_root=/raw7/cv1/hangchen2/misp2021_avsr/feature/misp2021_avsr/${type}_far_audio_gss
mkdir -p $out_root/log

mkdir -p $out_root/wav

echo $python_path
if [ $stage -le 0 ]; then
  echo "start gss"

  ~/anaconda3/bin/python find_wav.py $data_root $out_root/log gss Far -nj $nj
#   for n in `seq $nj`; do
#     cat <<-EOF > $out_root/log/gss.$n.sh
#     ~/anaconda3/bin/python run_gss.py $out_root/log/gss.$n.scp $data_root $out_root Far
# EOF
  ~/anaconda3/bin/python run_gss.py $out_root/log/gss.scp $data_root $out_root Far
# dlp submit -a hangchen2 -n gss${type} -d gss${type} -e $out_root/log/gss.$n.sh -i reg.deeplearning.cn/dlaas/cv_dist:0.1 -l $out_root/log/gss.$n.log --useGpu -g 1 -t PtJob -k TeslaM40
  # done
  echo "finish gss"
fi