# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow_temp.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(970, 773)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        self.upload_picture_button = QtWidgets.QPushButton(self.centralwidget)
        self.upload_picture_button.setObjectName("upload_picture_button")
        self.horizontalLayout_3.addWidget(self.upload_picture_button)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.label_original_pic = QtWidgets.QLabel(self.centralwidget)
        self.label_original_pic.setAlignment(QtCore.Qt.AlignCenter)
        self.label_original_pic.setObjectName("label_original_pic")
        self.verticalLayout_2.addWidget(self.label_original_pic)
        self.verticalLayout_2.setStretch(0, 2)
        self.verticalLayout_2.setStretch(1, 5)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout.addItem(spacerItem2)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem3)
        self.instance_button = QtWidgets.QPushButton(self.centralwidget)
        self.instance_button.setObjectName("instance_button")
        self.horizontalLayout_4.addWidget(self.instance_button)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem4)
        self.verticalLayout_4.addLayout(self.horizontalLayout_4)
        self.label_instance_pic = QtWidgets.QLabel(self.centralwidget)
        self.label_instance_pic.setAlignment(QtCore.Qt.AlignCenter)
        self.label_instance_pic.setObjectName("label_instance_pic")
        self.verticalLayout_4.addWidget(self.label_instance_pic)
        self.verticalLayout_4.setStretch(0, 2)
        self.verticalLayout_4.setStretch(1, 5)
        self.horizontalLayout.addLayout(self.verticalLayout_4)
        spacerItem5 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout.addItem(spacerItem5)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem6)
        self.defect_detection_button = QtWidgets.QPushButton(self.centralwidget)
        self.defect_detection_button.setObjectName("defect_detection_button")
        self.horizontalLayout_5.addWidget(self.defect_detection_button)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem7)
        self.verticalLayout_6.addLayout(self.horizontalLayout_5)
        self.label_defect_pic = QtWidgets.QLabel(self.centralwidget)
        self.label_defect_pic.setAlignment(QtCore.Qt.AlignCenter)
        self.label_defect_pic.setObjectName("label_defect_pic")
        self.verticalLayout_6.addWidget(self.label_defect_pic)
        self.verticalLayout_6.setStretch(0, 2)
        self.verticalLayout_6.setStretch(1, 5)
        self.horizontalLayout.addLayout(self.verticalLayout_6)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_remain = QtWidgets.QLabel(self.centralwidget)
        self.label_remain.setText("")
        self.label_remain.setObjectName("label_remain")
        self.horizontalLayout_7.addWidget(self.label_remain)
        spacerItem8 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout_7.addItem(spacerItem8)
        self.label_proportion_defect = QtWidgets.QLabel(self.centralwidget)
        self.label_proportion_defect.setObjectName("label_proportion_defect")
        self.horizontalLayout_7.addWidget(self.label_proportion_defect)
        spacerItem9 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout_7.addItem(spacerItem9)
        self.label_defect_and_number = QtWidgets.QLabel(self.centralwidget)
        self.label_defect_and_number.setObjectName("label_defect_and_number")
        self.horizontalLayout_7.addWidget(self.label_defect_and_number)
        self.horizontalLayout_7.setStretch(0, 7)
        self.horizontalLayout_7.setStretch(1, 1)
        self.horizontalLayout_7.setStretch(2, 7)
        self.horizontalLayout_7.setStretch(3, 1)
        self.horizontalLayout_7.setStretch(4, 7)
        self.verticalLayout.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem10 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem10)
        self.query_kg_button = QtWidgets.QPushButton(self.centralwidget)
        self.query_kg_button.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.query_kg_button.setObjectName("query_kg_button")
        self.horizontalLayout_2.addWidget(self.query_kg_button)
        spacerItem11 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout_2.addItem(spacerItem11)
        self.table_kg = QtWidgets.QTableWidget(self.centralwidget)
        self.table_kg.setObjectName("table_kg")
        self.table_kg.setColumnCount(0)
        self.table_kg.setRowCount(0)
        self.horizontalLayout_2.addWidget(self.table_kg)
        spacerItem12 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout_2.addItem(spacerItem12)
        self.save_form_button = QtWidgets.QPushButton(self.centralwidget)
        self.save_form_button.setObjectName("save_form_button")
        self.horizontalLayout_2.addWidget(self.save_form_button)
        spacerItem13 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem13)
        self.horizontalLayout_2.setStretch(1, 2)
        self.horizontalLayout_2.setStretch(3, 20)
        self.horizontalLayout_2.setStretch(4, 1)
        self.horizontalLayout_2.setStretch(5, 2)
        self.horizontalLayout_2.setStretch(6, 1)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.verticalLayout.setStretch(0, 6)
        self.verticalLayout.setStretch(2, 6)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.upload_picture_button.setText(_translate("MainWindow", "上传图片"))
        self.label_original_pic.setText(_translate("MainWindow", "显示实例分割图"))
        self.instance_button.setText(_translate("MainWindow", "进行实例分割"))
        self.label_instance_pic.setText(_translate("MainWindow", "显示实例分割图"))
        self.defect_detection_button.setText(_translate("MainWindow", "进行缺陷检测"))
        self.label_defect_pic.setText(_translate("MainWindow", "显示目标检测图"))
        self.label_proportion_defect.setText(_translate("MainWindow", "缺陷占比："))
        self.label_defect_and_number.setText(_translate("MainWindow", "缺陷类型：  缺陷个数："))
        self.query_kg_button.setText(_translate("MainWindow", "查询知识图谱"))
        self.save_form_button.setText(_translate("MainWindow", "保存表格"))
