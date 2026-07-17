# core/node.py

class Node:
    """ The Node class."""
    def __init__(self, value, parent=None, offspring=[]):
        self.parent = parent
        self.offspring = offspring
        self.value = value
        self.order = len(self.offspring)
        return

    def pr(self, show_pow=False):
        if self.offspring == []:
            return '%s' % self.value
        elif len(self.offspring) == 2:
            return '(%s %s %s)' % (self.offspring[0].pr(show_pow=show_pow),
                                   self.value,
                                   self.offspring[1].pr(show_pow=show_pow))
        else:
            if show_pow:
                return '%s(%s)' % (self.value,
                                   ','.join([o.pr(show_pow=show_pow)
                                             for o in self.offspring]))
            else:
                if self.value == 'pow2':
                    return '(%s ** 2)' % (
                        self.offspring[0].pr(show_pow=show_pow)
                    )
                elif self.value == 'pow3':
                    return '(%s ** 3)' % (
                        self.offspring[0].pr(show_pow=show_pow)
                    )
                else:
                    return '%s(%s)' % (
                        self.value,
                        ','.join([o.pr(show_pow=show_pow)
                                  for o in self.offspring])
                    )
