# First token represents "nothing"
#CHAR_VECTOR = "*abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CHAR_VECTOR = "*ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789`~!@#$%^&()-_=+[{}]\|;:',<.>/?\""
# CHAR_VECTOR = "*AÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬOÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢIÍÌỈĨỊUÚÙỦŨỤƯỨỪỬỮỰEÉÈẺẼẸÊẾỀỂỄỆYÝỲỶỸỴBCDĐƉÐFGHJKLMNPQRSTVWXZ0123456789~()-+;:'.,/_"

# Number of classes
NUM_CLASSES = len(CHAR_VECTOR) + 1
print('NUMBERBLASSS', NUM_CLASSES)
