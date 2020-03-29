
class IsClassMixin(object):
    _class_name_base = ''

    def __getattr__(self, attr_name):

        """
        Rather than having to do isinstance(token, SomeToken), this allows
        for token.is_some_token
        """
        if attr_name.startswith("is_"):
            cls_name = attr_name[3:].title().replace("_", "")
            if self._class_name_base:
                cls_name += self._class_name_base
            return self.__class__.__name__ == cls_name

        raise AttributeError("{} has no attribute {}".format(self, attr_name))
