# automatic release notes
from Tests.Utils.ResearchUtils import print_var
from Tests.Utils.TestsUtils import last_delivered_version
from Tests.Constants import DELIVERED

import pandas as pd
import os
import smtplib
from string import Template
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

SEND_FROM = 'moshe.caspi.python@gmail.com'
SEND_TO = 'moshe.caspi@neteera.com'
EMAIL_PWD = 'Neteera@cool'


def send_mail(text, title, files):
    message_template = Template(text)

    smtplib_server = smtplib.SMTP(host='smtp.gmail.com', port=587)
    smtplib_server.starttls()
    smtplib_server.login(SEND_FROM, EMAIL_PWD)

    multipart_message = MIMEMultipart()       # create a message

    # add in the actual person name to the message template
    msg = message_template.substitute(PERSON_NAME='User')
    print(msg)

    # setup the parameters of the message
    multipart_message['From'] = SEND_FROM
    multipart_message['To'] = SEND_TO
    multipart_message['Subject'] = title

    # add in the message body
    multipart_message.attach(MIMEText(msg, 'plain'))

    for path in files:
        part = MIMEBase('application', 'octet-stream')
        with open(path, 'rb') as file:
            part.set_payload(file.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(path)}')
        multipart_message.attach(part)

    smtplib_server.send_message(multipart_message)
    del multipart_message

    smtplib_server.quit()


def get_changes_path(ver_name):
    for file in os.listdir(os.path.join(DELIVERED, ver_name, ver_name)):
        if 'changes' in file:
            return file


def mail_template(ver):
    file_path = get_changes_path(ver)
    ver_changes = pd.read_excel(file_path)
    number_changes = ver_changes['Version'][ver_changes['Version'].isna()].index[1] - 1
    changes = ver_changes['Essence of change'][:number_changes]
    changes_list_str = '\n'.join([f'\t{i+1}. {change}' for i, change in changes.items()])

    greatest_changes_pptx = []
    compare_excel = []
    stats = os.path.join(DELIVERED, ver, 'stats')
    for bench in set(os.listdir(stats)) - {'ec_benchmark', 'nwh', 'bugs'}:
        if os.path.isdir(bench):
            bench_files = os.listdir(os.path.join(stats, bench))
            greatest_changes_pptx += [os.path.join(stats, bench, file) for file in bench_files
                                      if 'greatest_change' in file]
            compare_excel += [os.path.join(stats, bench, file) for file in bench_files
                              if 'compare' in file and file.endswith('xlsx')]
    attachment_size = sum([os.path.getsize(x) * 1e-6 for x in greatest_changes_pptx + compare_excel])
    print_var(attachment_size)
    send_mail(fr"""Hi all,
     
    {ver} is released with the following:
    {changes_list_str}
    
    Statistics, Plots, and code:
      N:\neteeraVirtualServer\DELIVERED\Algo\{ver}\stats
      
    Notes:   
        1. 
        
    Regards,
        Moshe
    """, ver, greatest_changes_pptx+compare_excel)


if __name__ == '__main__':
    mail_template(last_delivered_version())
