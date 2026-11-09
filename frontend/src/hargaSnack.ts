// Data harga varian gramasi per snack, dibaca dari data/Harga_snack.csv (hasil simpan Excel).
// Kolom CSV: Nama Lengkap Produk, Pilihan Gramasi Resmi, Format Kemasan,
//   8 kolom harga toko, Estimasi Biaya (kolom terakhir).
// Nama produk yang kosong berarti melanjutkan produk di baris atasnya.
// Harga yang dipakai untuk kalkulasi di halaman Price Estimation adalah kolom Estimasi Biaya.
import csvHarga from './data/Harga_snack.csv?raw'

export type DataHarga = {
  kemasan: string // contoh: Pouch Reguler
  harga: number[] // harga per toko
  rata: number // Estimasi Biaya (dipakai untuk kalkulasi)
}

export type DataSnack = {
  label: string
  harga: Record<number, DataHarga> // gramasi (g) -> data harga
}
// Label tampilan per class model.
const LABEL: Record<string, string> = {
  japota_madu: 'Japota Potato Chips Happy Honey Butter',
  japota_rumput_laut: 'Japota Potato Chips Umami Japanese Seaweed',
  potabee_ayam_bakar: 'Potabee Potato Chips Ayam Bakar',
  potabee_spicy: 'Potabee Potato Chips Spicy BBQ',
  potabee_sapi_pangang: 'Potabee Potato Chips Sapi Panggang',
  potabee_rumput_laut: 'Potabee Potato Chips Rumput Laut Panggang',
  chitato_lite_salmon: 'Chitato Lite Salmon Teriyaki',
  chitato_lite_ayam_bawang: 'Chitato Lite Ayam Bawang',
  chitato_lite_rumput_laut: 'Chitato Lite Rumput Laut (Japanese Seaweed)',
  chitato_lite_saus_bawang: 'Chitato Lite Saus Krim Bawang (Sour Cream)',
  chitato_aburi: 'Chitato Lite Aburi Seaweed',
  chitato_sapi_panggang: 'Chitato Sapi Panggang (Beef Barbeque)',
  chitato_bumbu_sapi: 'Chitato Sapi Bumbu Bakar',
  qtela_ori: 'Qtela Singkong Original',
  qtela_balado: 'Qtela Singkong Balado',
}

// Mengubah teks Rupiah ("11.400", "11 900", "11570") menjadi angka; titik/spasi diabaikan.
function angkaRupiah(teks: string): number {
  return Number(teks.replace(/\D/g, ''))
}

// Mengubah isi CSV menjadi tabel harga per snack (key = nama class model) dan gramasi.
function bacaCsv(isi: string): Record<string, DataSnack> {
  const kunciDariLabel = Object.fromEntries(Object.entries(LABEL).map(([k, v]) => [v, k]))
  const hasil: Record<string, DataSnack> = {}
  let nama = ''
  isi.split(/\r?\n/).forEach(baris => {
    const kolom = baris.split(',').map(k => k.trim())
    const gramasi = parseInt(kolom[1])
    // Lewati baris kosong dan header (gramasi bukan angka).
    if (kolom.length < 5 || !/^\d+\s*g$/i.test(kolom[1] ?? '')) return
    if (kolom[0]) nama = kolom[0]
    const kunci = kunciDariLabel[nama]
    if (!kunci) {
      console.warn('Nama produk di Harga_snack.csv tidak dikenal:', nama)
      return
    }
    const angka = kolom.slice(3).filter(k => k !== '').map(angkaRupiah)
    hasil[kunci] ??= { label: nama, harga: {} }
    hasil[kunci].harga[gramasi] = {
      kemasan: kolom[2],
      harga: angka.slice(0, -1),
      rata: angka[angka.length - 1],
    }
  })
  return hasil
}
export const HARGA_SNACK = bacaCsv(csvHarga)
// Mengembalikan label tampilan, atau nama class bila tidak ada di tabel harga.
export function labelSnack(nama: string): string {
  return HARGA_SNACK[nama]?.label ?? nama
}

// Mengembalikan daftar gramasi valid (urut naik) untuk sebuah snack.
export function daftarGramasi(nama: string): number[] {
  return Object.keys(HARGA_SNACK[nama]?.harga ?? {}).map(Number).sort((a, b) => a - b)
}

// Mengembalikan nama format kemasan satu varian (mis. Pouch Reguler).
export function kemasanVarian(nama: string, gramasi: number): string {
  return HARGA_SNACK[nama]?.harga[gramasi]?.kemasan ?? ''
}

// Mengembalikan harga satu varian, 0 bila tidak ditemukan.
export function hargaVarian(nama: string, gramasi: number): number {
  return HARGA_SNACK[nama]?.harga[gramasi]?.rata ?? 0
}

// Memformat angka menjadi mata uang Rupiah.
export function formatRupiah(nilai: number): string {
  return 'Rp ' + nilai.toLocaleString('id-ID')
}
