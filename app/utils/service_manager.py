class ServiceManager:
    _instances = {}

    @classmethod
    def get_service(self, service_class):
        if service_class not in self._instances:
            self._instances[service_class] = service_class()
        return self._instances[service_class]

    @classmethod
    def reset_service(self, service_class):
        if service_class in self._instances:
            del self._instances[service_class]