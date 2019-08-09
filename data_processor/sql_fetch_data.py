import  cx_Oracle as cx

import pandas.io.sql as sql
import os
import sys
# sys.path.append(".")
# print(sys.path)

os.environ['NLS_LANG'] = 'AMERICAN_AMERICA.AL32UTF8'

ip = '172.16.20.21'
port = 1521
db = 'smsdb'
username = "nlp"
pwd = "jPKNuAW8y8nzvXbG"

dsn = cx.makedsn(ip, port, db)
connection = cx.connect(username, pwd, dsn)

print("oracle版本：", connection.version)
cursor = connection.cursor()



meta_data = sql.read_sql("""select account,content,state,key_word,reject_reason,to_char(audit_time,'yyyymmdd') as audit_time,sys_user from smsdb.HTTP_BATCH_SMS_A_20190801  where  state=4 and AUDIT_TIME is not null  group by account,content,state,key_word,reject_reason,to_char(audit_time,'yyyymmdd'),sys_user""", connection)

print(meta_data)
meta_data.to_csv('pos-sql.csv')
# meta_data.dropna(axis=0, how='any', inplace=True)
#
# meta_data.drop_duplicates()

cursor.close()

connection.close()
