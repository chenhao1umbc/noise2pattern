#This is the RSA crypto scheme (public key and private key)
#@Hao: (1) pip3 install pycryptodome
#      (2) key size: 1024, 2048, 3072, 4096
       
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import binascii

#key generation
keyPair = RSA.generate(3072)

pubKey = keyPair.publickey()

pubKeyPEM = pubKey.exportKey()

privKeyPEM = keyPair.exportKey()



#encryption  
msg = b'A message for encryption' #you can input the message here (image size should be equal to the length of the key length!!!!!!!)

encryptor = PKCS1_OAEP.new(pubKey) #using public key to encrypt

encrypted = encryptor.encrypt(msg)

print("Encrypted:", binascii.hexlify(encrypted))

#decryption
decryptor = PKCS1_OAEP.new(keyPair) #using private key to decrypt

decrypted = decryptor.decrypt(encrypted)

print('Decrypted:', decrypted)
