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
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1J4tKvzC5v1QcO6XEeyInWBEV6P0idCeL' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1J4tKvzC5v1QcO6XEeyInWBEV6P0idCeL" -O vggish_vatex.zip && rm -rf /tmp/cookies.txt -q --show-progress
echo "Downloading i3d vatex features"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1U2PZTnVAA8UWIqZRicUqITuUNOl1jI9x' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1U2PZTnVAA8UWIqZRicUqITuUNOl1jI9x" -O i3d_vatex.zip && rm -rf /tmp/cookies.txt -q --show-progress
cd ../
echo "Downloading i3d action features"
cd data/
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
i3d_vatex_md5=($(md5sum ./data/i3d_vatex.zip))
if [ "$i3d_vatex_md5" == "aaff867b2cd4aec9eef5f9d0b226d525" ]; then
    echo "OK: i3d vatex features"
else
    echo "ERROR: .zip file with i3d vatex features is corrupted"
    exit 1
fi

vggish_vatex_md5=($(md5sum ./data/vggish_vatex.zip))
if [ "$vggish_vatex_md5" == "f57201092e1d90b6bf35c55e575c6994" ]; then
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
unzip -q vggish_vatex.zip
echo "Unpacking i3d vatex features"
unzip -q i3d_vatex.zip
echo "Done"
