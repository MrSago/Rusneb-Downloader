import random
import httpx
import asyncio

from src.config.config import Config
from src.models.parse_request import ParseRequest
from src.models.page_data import PageData
from src.client.client_manager import ClientManager
from src.utils.log_manager import log_manager


class Downloader:
    """Класс для загрузки документов с сайта Rusneb."""

    DOWNLOAD_URL = f"https://rusneb.ru/local/tools/exalead/getFiles.php"

    def __init__(
        self,
        request: ParseRequest,
        client_manager: ClientManager,
        page_data: PageData,
        num_workers: int = Config.DEFAULT_DOWNLOADER_WORKERS,
        max_retries: int = Config.DEFAULT_DOWNLOADER_RETRIES,
    ):
        """
        Инициализация класса Downloader.

        Args:
            request (ParseRequest): Запрос для загрузки документов.
            client_manager (ClientManager): Менеджер HTTP-клиентов.
            page_data (PageData): Данные страницы с информацией о загрузке.
            num_workers (int): Количество воркеров для загрузки.
        """

        self.request = request
        self.client_manager = client_manager
        self.page_data = page_data
        self.num_workers = num_workers
        self.max_retries = max_retries

        self.stop_event = asyncio.Event()
        self.logger = log_manager.get_logger(__name__)

        self.save_directory = Config.RESULT_DIR / request.query / "downloads"
        self.save_directory.mkdir(parents=True, exist_ok=True)

    async def run(self) -> None:
        """Запуск загрузчика документов с использованием асинхронных воркеров."""

        workers = [asyncio.create_task(self.worker(i)) for i in range(self.num_workers)]
        pending = set(workers)

        try:
            done, pending = await asyncio.wait(
                workers, return_when=asyncio.FIRST_COMPLETED
            )

        except asyncio.CancelledError:
            self.logger.warning("Загрузка была отменена пользователем")
        except Exception as e:
            self.logger.exception(f"Вызвано исключение в воркерах: {e}", exc_info=True)

        finally:
            if not self.stop_event.is_set():
                self.stop_event.set()
                self.logger.info("Ожидание завершения всех воркеров...")

            for worker in pending:
                if not worker.done():
                    worker.cancel()

            await asyncio.gather(*pending, return_exceptions=True)

    async def worker(self, worker_id: int) -> None:
        """Асинхронный воркер для загрузки документов."""

        self.logger.info(f"Воркер {worker_id} запущен для загрузки документов")

        client = await self.client_manager.pop_client()

        async with client:
            while not self.stop_event.is_set():
                success = False
                file_id = None
                try:
                    async with self.page_data.protected_access() as data:
                        if not data.download_queue:
                            self.logger.info(
                                "Очередь загрузки пуста, ожидание обновлений..."
                            )
                            await asyncio.sleep(10)
                            continue
                        file_id = data.download_queue.popleft()
                        if file_id in data.downloaded:
                            continue
                    success = await self._download_file(client, file_id)

                except asyncio.CancelledError:
                    self.logger.warning(f"Воркер {worker_id} отменен")
                except Exception as e:
                    self.logger.exception(
                        f"Ошибка в воркере {worker_id} для файла {file_id}: {e}",
                        exc_info=True,
                    )

                finally:
                    if file_id:
                        async with self.page_data.protected_access() as data:
                            if success:
                                data.downloaded.add(file_id)
                                self.logger.info(f"Файл {file_id} успешно загружен")
                            else:
                                self.logger.warning(
                                    f"Не удалось загрузить {file_id}, повторное добавление в очередь"
                                )
                                data.download_queue.append(file_id)
                    await asyncio.sleep(random.uniform(1, 3))

    async def _download_file(self, client: httpx.AsyncClient, file_id: str) -> bool:
        """
        Загрузка файла по идентификатору с механизмом повторных попыток.

        Args:
            client (httpx.AsyncClient): Асинхронный HTTP-клиент.
            file_id (str): Идентификатор файла для загрузки.

        Returns:
            bool: True, если загрузка прошла успешно, иначе False.
        """

        self.logger.info(f"Загрузка файла {file_id}")

        params = {"book_id": file_id, "doc_type": "pdf"}

        for attempt in range(1, self.max_retries + 1):
            self.logger.info(
                f"Попытка {attempt}/{self.max_retries} загрузки файла {file_id}"
            )
            try:
                async with client.stream(
                    "GET", self.DOWNLOAD_URL, params=params, timeout=120.0
                ) as response:
                    if response.status_code != 200:
                        self.logger.warning(
                            f"Ошибка загрузки файла {file_id}: статус {response.status_code}"
                        )
                        if attempt < self.max_retries:
                            wait_time = 2**attempt
                            self.logger.info(
                                f"Повторная попытка {attempt} через {wait_time} сек..."
                            )
                            await asyncio.sleep(wait_time)
                        continue

                    content_type = response.headers.get("Content-Type")
                    if content_type != "application/pdf":
                        self.logger.warning(
                            f"Неверный тип содержимого для {file_id}: {content_type}, ожидается application/pdf"
                        )
                        return False

                    filename = (
                        response.headers.get("Content-Disposition", "")
                        .split("filename=")[-1]
                        .strip('"')
                    )
                    filename = file_id if not filename else filename

                    full_path = self.save_directory / f"{filename}.pdf"

                    with open(full_path, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            f.write(chunk)

                    return True

            except httpx.ReadError:
                if attempt < self.max_retries:
                    wait_time = 2**attempt
                    self.logger.warning(
                        f"Ошибка чтения при загрузке {file_id} (попытка {attempt}/{self.max_retries}). "
                        f"Повторная попытка через {wait_time} сек..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(
                        f"Не удалось загрузить {file_id} после {self.max_retries} попыток"
                    )
            except httpx.TimeoutException:
                if attempt < self.max_retries:
                    wait_time = 2**attempt
                    self.logger.warning(
                        f"Тайм-аут при загрузке {file_id} (попытка {attempt}/{self.max_retries}). "
                        f"Повторная попытка через {wait_time} сек..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(
                        f"Тайм-аут при загрузке {file_id} после {self.max_retries} попыток"
                    )
            except Exception as e:
                self.logger.exception(
                    f"Вызвано исключение для {file_id} (попытка {attempt}/{self.max_retries}): {e}",
                    exc_info=True,
                )
                if attempt < self.max_retries:
                    wait_time = 2**attempt
                    self.logger.info(f"Повторная попытка через {wait_time} сек...")
                    await asyncio.sleep(wait_time)

        return False
