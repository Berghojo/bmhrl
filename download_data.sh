# checking if wget is installed on a computer
if ! command -v wget &> /dev/null
then
    echo "wget: command not found"
    echo ""
    echo "wget command could not be found on your computer. Please, install it first."
    echo "If you cannot/dontwantto install wget, you may try to download the features manually."
    echo "You may find the links and correct paths in this file."
    echo "Make sure to check the md5 sums after manual download:"
    echo "./data/i3d_25fps_stack64step64_2stream_npy.zip    d7266e440f8c616acbc0d8aaa4a336dc"
    echo "./data/vggish_npy.zip    9a654ad785e801aceb70af2a5e1cffbe"
    echo "./.vector_cache/glove.840B.300d.zip    2ffafcc9f9ae46fc8c95f32372976137"
    exit
fi

echo "Downloading vggish vatex features"
cd data/
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=10memDR6jcmq3pELZ9GlYLtLCTT_xAqz7' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=10memDR6jcmq3pELZ9GlYLtLCTT_xAqz7" -O vggish_vatex.zip && rm -rf /tmp/cookies.txt -q --show-progress
echo "Downloading i3d vatex features"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1yF7isFK-_NeKE-krdlI-YbrKi3DyjDts' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1yF7isFK-_NeKE-krdlI-YbrKi3DyjDts" -O i3d_vatex.zip && rm -rf /tmp/cookies.txt -q --show-progress

echo "Downloading i3d msrvtt features"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=10viD1S-lT7PsvAoUz4qxzsDxVJFe-WIS' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=10viD1S-lT7PsvAoUz4qxzsDxVJFe-WIS" -O i3d_msrvtt.zip && rm -rf /tmp/cookies.txt -q --show-progress
echo "Downloading vggish msrvtt features"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1mfF11gwTugwPXbxAJbI06271-G_2MDOM' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1mfF11gwTugwPXbxAJbI06271-G_2MDOM" -O vggish_msrvtt.zip && rm -rf /tmp/cookies.txt -q --show-progress
echo "Downloading i3d action features"
wget https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/bmt/i3d_25fps_stack64step64_2stream_npy.zip -q --show-progress
echo "Downloading vggish action features"
wget https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/bmt/vggish_npy.zip -q --show-progress
cd ../

echo "Downloading GloVe embeddings"
mkdir .vector_cache
cd .vector_cache
wget https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/bmt/glove.840B.300d.zip -q --show-progress
cd ../

echo "Checking for correctness of the downloaded files"
i3d_msr_md5=($(md5sum ./data/i3d_msrvtt.zip))
if [ "$i3d_msr_md5" == "5ec6a58ce2b2f167612884cddeaf88ec" ]; then
    echo "OK: i3d msrvtt features"
else
    echo "ERROR: .zip file with i3d msrvtt features is corrupted"
    exit 1
fi

vggish_msr_md5=($(md5sum ./data/vggish_msrvtt.zip))
if [ "$vggish_msr_md5" == "997b32e215fee6cea6616d3f90559ea2" ]; then
    echo "OK: vggish msrvtt features"
else
    echo "ERROR: .zip file with vggish msrvtt features is corrupted"
    exit 1
fi

i3d_vatex_md5=($(md5sum ./data/i3d_vatex.zip))
if [ "$i3d_vatex_md5" == "ad708eac086b5aadf5df47990382b53b" ]; then
    echo "OK: i3d vatex features"
else
    echo "ERROR: .zip file with i3d vatex features is corrupted"
    exit 1
fi

vggish_vatex_md5=($(md5sum ./data/vggish_vatex.zip))
if [ "$vggish_vatex_md5" == "e5a8603d7b737310d9b27e4e8c48be80" ]; then
    echo "OK: vggish vatex features"
else
    echo "ERROR: .zip file with vggish vatex features is corrupted"
    exit 1
fi

i3d_md5=($(md5sum ./data/i3d_25fps_stack64step64_2stream_npy.zip))
if [ "$i3d_md5" == "d7266e440f8c616acbc0d8aaa4a336dc" ]; then
    echo "OK: i3d features"
else
    echo "ERROR: .zip file with i3d features is corrupted"
    exit 1
fi

vggish_md5=($(md5sum ./data/vggish_npy.zip))
if [ "$vggish_md5" == "9a654ad785e801aceb70af2a5e1cffbe" ]; then
    echo "OK: vggish features"
else
    echo "ERROR: .zip file with vggish features is corrupted"
    exit 1
fi

glove_md5=($(md5sum ./.vector_cache/glove.840B.300d.zip))
if [ "$glove_md5" == "2ffafcc9f9ae46fc8c95f32372976137" ]; then
    echo "OK: glove embeddings"
else
    echo "ERROR: .zip file with glove embeddings is corrupted"
    exit 1
fi

echo "Unpacking i3d (~1 min)"
cd ./data

unzip -q i3d_25fps_stack64step64_2stream_npy.zip
echo "Unpacking vggish features"
unzip -q vggish_npy.zip
echo "Unpacking vggish vatex features"
unzip -q vggish_vatex.zip -d vatex
echo "Unpacking i3d vatex features"
unzip -q i3d_vatex.zip -d vatex
echo "Unpacking vggish msrvtt features"
unzip -q vggish_msrvtt.zip -d msrvtt
echo "Unpacking i3d msrvtt features"
unzip -q i3d_msrvtt.zip -d msrvtt
echo "Done"
