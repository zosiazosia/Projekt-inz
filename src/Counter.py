from enum import Enum


class direction(Enum):
    LEFT = 'left'
    RIGHT = 'right'

class Counter:
    def __init__(self, inDir):
        self.reident_in = 0
        self.reident_out = 0
        self.are_inside = 0
        self.inDirection = inDir  # 'left or 'right'
        self.error_information = " "
        self.regular_in = 0
        self.regular_out = 0

    def reid_in(self):
        self.reident_in += 1

    def reid_out(self):
        self.reident_out += 1

    def getReidentInString(self):
        return str(self.reident_in)

    def getReidentOutString(self):
        return str(self.reident_out)

    def getAreInsideString(self):
        return str(self.are_inside)

    def getInDirectionString(self):
        return self.inDirection

    def getRegularInString(self):
        return str(self.regular_in)

    def getRegularOutString(self):
        return str(self.regular_out)

    def generate_report(self, type):
        info = "came_in: " + self.getRegularInString() + ", came_out: " + self.getRegularOutString() + ", reid_in: " + \
               self.getReidentInString() + ", reid_out: " + self.getReidentOutString() + ", inside: " + self.getAreInsideString()

        report_eng = "Regular counter information: \n %s has come in and %s people has come out. \n" \
                     "Intelligent counter information: \n " \
                 "%s has been reidentified coming in and  %  has been reidentified coming out. " \
                 "Currently there are %s people inside. " \
                     % (self.getRegularInString(), self.getRegularOutString(), self.getReidentInString(),
                        self.getReidentOutString(), self.getAreInsideString())

        report_pl = "Tradycyjny licznik osób: \n %s osób weszło oraz %s osób wyszło. \n\n" \
                    "Inteligentny licznik osób: \n " \
                    "%s osób zostało zreidentyfikowanych wchodząc oraz %s osób zostało zreidentyfikowanych wychodząc. " \
                    "Obecnie w środku znajduje się %s osób. " \
                    % (self.getRegularInString(), self.getRegularOutString(), self.getReidentInString(),
                       self.getReidentOutString(), self.getAreInsideString())

        if type == 'eng':
            report = report_eng
        else:
            report = report_pl

        return report

    def increase_regular_in(self):
        self.regular_in += 1
        self.are_inside += 1

    def increase_regular_out(self):
        self.regular_out += 1
        if not self.are_inside == 0:
            self.are_inside -= 1
